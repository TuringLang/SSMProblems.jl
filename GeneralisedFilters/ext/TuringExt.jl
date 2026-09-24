module TuringExt

using GeneralisedFilters
import GeneralisedFilters:
    SSMTrajectory, _csmc_sample, _get_inner_filter, _state_dim, _flatten_trajectory
using AbstractMCMC: AbstractMCMC
using Bijectors: Bijectors
using Bijectors.VectorBijectors: TypedIdentity
import Bijectors: bijector
using Distributions: Distributions
using DynamicPPL: DynamicPPL
using Random: AbstractRNG
using Turing: Turing

## BIJECTORS INTEGRATION #######################################################################

bijector(::SSMTrajectory) = identity

Bijectors.VectorBijectors.to_linked_vec(::SSMTrajectory) = TypedIdentity()
Bijectors.VectorBijectors.from_linked_vec(::SSMTrajectory) = TypedIdentity()
Bijectors.VectorBijectors.linked_vec_length(d::SSMTrajectory) = length(d)

## CSMC SAMPLING CONTEXT #######################################################################

"""
    CSMCContext{<:AbstractRNG,<:ConditionalSMC}

A DynamicPPL leaf context that intercepts `x ~ SSMTrajectory(...)` and, rather than
evaluating the prior-path log-density, runs conditional SMC to draw a new trajectory.
"""
# The context exists before encountering the trajectory distribution. Its references can
# hold different trajectory types; _csmc_sample specialises the numerical loop on the model.
struct CSMCContext{RT<:AbstractRNG,FT<:ConditionalSMC} <: DynamicPPL.AbstractContext
    rng::RT
    algo::FT
    ref_traj::Ref{Any}
    sampled_traj::Ref{Any}
end

function CSMCContext(rng::AbstractRNG, algo::ConditionalSMC; ref=nothing)
    return CSMCContext(rng, algo, Ref{Any}(ref), Ref{Any}(nothing))
end

function conditional_smc(ctx::CSMCContext, dist::SSMTrajectory)
    _get_inner_filter(ctx.algo.pf) == dist.af ||
        throw(ArgumentError("SSMTrajectory analytical filter must match the CSMC filter"))
    ctx.sampled_traj[] === nothing || throw(
        ArgumentError(
            "ParticleGibbs currently supports exactly one SSMTrajectory variable"
        ),
    )
    trajectory, _ = _csmc_sample(
        ctx.rng, dist.model, ctx.algo, dist.observations, ctx.ref_traj[]
    )
    return trajectory
end

function flatten_trajectory(ctx::CSMCContext, dist::SSMTrajectory)
    trajectory = ctx.sampled_traj[]
    return _flatten_trajectory(trajectory, length(dist.observations), _state_dim(dist))
end

function DynamicPPL.tilde_assume!!(
    ctx::CSMCContext,
    dist::SSMTrajectory,
    vn::DynamicPPL.VarName,
    template,
    vi::DynamicPPL.AbstractVarInfo,
)
    ctx.sampled_traj[] = conditional_smc(ctx, dist)
    x_flat = flatten_trajectory(ctx, dist)
    vi = DynamicPPL.setindex_internal!!(vi, x_flat, vn)
    vi = DynamicPPL.accumulate_assume!!(
        vi, x_flat, x_flat, zero(Float64), vn, dist, template
    )
    return x_flat, vi
end

function DynamicPPL.tilde_assume!!(
    ::CSMCContext,
    dist::Distributions.Distribution,
    vn::DynamicPPL.VarName,
    template::Any,
    vi::DynamicPPL.AbstractVarInfo,
)
    return DynamicPPL.tilde_assume!!(DynamicPPL.DefaultContext(), dist, vn, template, vi)
end

function DynamicPPL.tilde_observe!!(
    ::CSMCContext,
    right::Distributions.Distribution,
    left,
    vn::Union{DynamicPPL.VarName,Nothing},
    template::Any,
    vi::DynamicPPL.AbstractVarInfo,
)
    return DynamicPPL.tilde_observe!!(
        DynamicPPL.DefaultContext(), right, left, vn, template, vi
    )
end

## TRAJECTORY VNT ACCUMULATOR ##################################################################

const TRAJ_ACCUMULATOR = :StateTrajectory

_collect_traj(val, _, _, _, ::SSMTrajectory) = val
_collect_traj(_, _, _, _, _) = DynamicPPL.DoNotAccumulate()

TrajectoryVNTAccumulator() = DynamicPPL.VNTAccumulator{TRAJ_ACCUMULATOR}(_collect_traj)

function get_trajectory(vi::DynamicPPL.AbstractVarInfo)
    return DynamicPPL.getacc(vi, Val(TRAJ_ACCUMULATOR)).values
end

## TURING STATE ################################################################################

struct ParticleGibbsTuringState{VIT,TT,PS,PT}
    vi::VIT
    trajectory::TT
    param_state::PS
    θ::PT
end

## ABSTRACTMCMC INTERFACE ######################################################################

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    pg::ParticleGibbs;
    initial_params=nothing,
    kwargs...,
)
    (initial_params === nothing || initial_params isa DynamicPPL.InitFromPrior) || throw(
        ArgumentError(
            "Turing ParticleGibbs initial_params is not yet supported; condition fixed values in the model",
        ),
    )
    # 1. Sample all variables from prior
    vi = DynamicPPL.setacc!!(DynamicPPL.VarInfo(rng, model), TrajectoryVNTAccumulator())

    # 2. Unconditional CSMC sweep
    ctx = CSMCContext(rng, pg.csmc)
    _, vi = DynamicPPL.evaluate_nowarn!!(DynamicPPL.setleafcontext(model, ctx), vi)
    trajectory = ctx.sampled_traj[]
    trajectory === nothing &&
        throw(ArgumentError("ParticleGibbs requires exactly one SSMTrajectory variable"))
    vnt_traj = get_trajectory(vi)

    # 3. Condition on trajectory
    cond_model = model | vnt_traj
    θ = DynamicPPL.subset(vi, Base.filter(vn -> !(vn in keys(vnt_traj)), keys(vi)))
    θ = DynamicPPL.link!!(θ, cond_model)
    ldf = DynamicPPL.LogDensityFunction(
        cond_model, DynamicPPL.getlogjoint_internal, θ; adtype=pg.adtype
    )
    _, param_state = AbstractMCMC.step(
        rng, AbstractMCMC.LogDensityModel(ldf), pg.param; initial_params=θ[:], kwargs...
    )

    # 4. Update VarInfo with new parameters
    θ_new = AbstractMCMC.getparams(param_state)
    vi = merge(vi, DynamicPPL.unflatten!!(θ, θ_new))

    # 5. Conditional CSMC with updated parameters
    vi = DynamicPPL.setacc!!(vi, TrajectoryVNTAccumulator())
    ctx_next = CSMCContext(rng, pg.csmc; ref=trajectory)
    _, vi = DynamicPPL.evaluate_nowarn!!(DynamicPPL.setleafcontext(model, ctx_next), vi)
    trajectory_new = ctx_next.sampled_traj[]

    transition = DynamicPPL.ParamsWithStats(
        DynamicPPL.InitFromParams(DynamicPPL.get_values(vi), nothing),
        model,
        AbstractMCMC.getstats(param_state),
    )
    state = ParticleGibbsTuringState(vi, trajectory_new, param_state, θ)
    return transition, state
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    pg::ParticleGibbs,
    state::ParticleGibbsTuringState;
    kwargs...,
)
    # 1. Condition on current trajectory via stored VNT
    vi = state.vi
    cond_model = model | get_trajectory(vi)

    # 2. Rebuild preparation and refresh cached density/gradient for the changed target.
    ldf = DynamicPPL.LogDensityFunction(
        cond_model, DynamicPPL.getlogjoint_internal, state.θ; adtype=pg.adtype
    )
    _, param_state = AbstractMCMC.step(
        rng,
        AbstractMCMC.LogDensityModel(ldf),
        pg.param,
        AbstractMCMC.setparams!!(
            AbstractMCMC.LogDensityModel(ldf),
            state.param_state,
            AbstractMCMC.getparams(state.param_state),
        );
        kwargs...,
    )

    # 3. Update VarInfo with new parameters
    θ_new = AbstractMCMC.getparams(param_state)
    vi = merge(vi, DynamicPPL.unflatten!!(state.θ, θ_new))

    # 4. Conditional CSMC with updated parameters
    vi = DynamicPPL.setacc!!(vi, TrajectoryVNTAccumulator())
    ctx = CSMCContext(rng, pg.csmc; ref=state.trajectory)
    _, vi = DynamicPPL.evaluate_nowarn!!(DynamicPPL.setleafcontext(model, ctx), vi)
    trajectory_new = ctx.sampled_traj[]

    transition = DynamicPPL.ParamsWithStats(
        DynamicPPL.InitFromParams(DynamicPPL.get_values(vi), nothing),
        model,
        AbstractMCMC.getstats(param_state),
    )
    new_state = ParticleGibbsTuringState(vi, trajectory_new, param_state, state.θ)
    return transition, new_state
end
end
