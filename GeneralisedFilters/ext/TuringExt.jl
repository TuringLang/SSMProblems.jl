module TuringExt

using GeneralisedFilters
import GeneralisedFilters:
    SSMTrajectory,
    _csmc_sample,
    _build_chains,
    _get_inner_filter,
    _outer_trajectory,
    _state_dim,
    _flatten_trajectory,
    ParticleGibbsTransition
using AbstractMCMC: AbstractMCMC
using Bijectors: Bijectors
using Bijectors.VectorBijectors: TypedIdentity
import Bijectors: bijector
import DifferentiationInterface as DI
using Distributions: Distributions
using DynamicPPL: DynamicPPL
using LinearAlgebra: PosDefException
using LogDensityProblems: LogDensityProblems
using MCMCChains: MCMCChains
using Random: AbstractRNG
using SSMProblems
using Turing: Turing

# include("dppl_csmc.jl")

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
struct CSMCContext{RT<:AbstractRNG,FT<:ConditionalSMC} <: DynamicPPL.AbstractContext
    rng::RT
    algo::FT
    ref_traj::Ref{Any}
    sampled_traj::Ref{Any}
end

# TODO: remove Ref{Any}
function CSMCContext(rng::AbstractRNG, algo::ConditionalSMC; ref=nothing)
    return CSMCContext(rng, algo, Ref{Any}(ref), Ref{Any}(nothing))
end

function conditional_smc(ctx::CSMCContext, dist::SSMTrajectory)
    trajectory, _ = _csmc_sample(
        ctx.rng, dist.model, ctx.algo, dist.observations, ctx.ref_traj[]
    )
    return trajectory
end

function flatten_trajectory(ctx::CSMCContext, dist::SSMTrajectory)
    af = _get_inner_filter(ctx.algo.pf)
    trajectory = ctx.sampled_traj[]
    return _flatten_trajectory(
        _outer_trajectory(trajectory, af), length(dist.observations), _state_dim(dist)
    )
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

## CSMC ACCUMULATOR ############################################################################

# TODO: replace the sampling context with one of these eventually...
struct CSMCFunctor{RT<:AbstractRNG,FT<:ConditionalSMC}
    rng::RT
    algo::FT
    ref::Ref{Any}
end

function CSMCFunctor(rng::AbstractRNG, algo::ConditionalSMC)
    return CSMCFunctor(rng, algo, Ref{Any}(nothing))
end

set_reference!(f::CSMCFunctor, traj) = (f.ref[] = traj)

function (f::CSMCFunctor)(_, _, _, _, dist::SSMTrajectory)
    trajectory, _ = _csmc_sample(f.rng, dist.model, f.algo, dist.observations, f.ref[])
    return trajectory
end

(::CSMCFunctor)(_, _, _, _, _) = DynamicPPL.DoNotAccumulate()

const CSMC_TRAJECTORY = :CSMCTrajectory

function csmc_accumulator(rng::AbstractRNG, algo::ConditionalSMC; ref=nothing)
    f = CSMCFunctor(rng, algo, Ref{Any}(ref))
    return DynamicPPL.VNTAccumulator{CSMC_TRAJECTORY}(f), f
end

function _get_csmc_trajectories(vi::DynamicPPL.AbstractVarInfo)
    return DynamicPPL.getacc(vi, Val(CSMC_TRAJECTORY)).values
end

## TRAJECTORY VNT ACCUMULATOR ##################################################################

const TRAJ_ACCUMULATOR = :StateTrajectory

_collect_traj(val, _, _, _, ::SSMTrajectory) = val
_collect_traj(_, _, _, _, _) = DynamicPPL.DoNotAccumulate()

TrajectoryVNTAccumulator() = DynamicPPL.VNTAccumulator{TRAJ_ACCUMULATOR}(_collect_traj)

function get_trajectory(vi::DynamicPPL.AbstractVarInfo)
    return DynamicPPL.getacc(vi, Val(TRAJ_ACCUMULATOR)).values
end

## CACHED PREP LDF #############################################################################

"""
    CachedPrepLDF

Wrapper that reuses the AD preparation (`_adprep`) from an existing `LogDensityFunction`
while evaluating against a different conditioned model. This avoids calling
`prepare_gradient` every MCMC iteration when only the conditioned trajectory changes.
"""
struct CachedPrepLDF{Tlink,L<:DynamicPPL.LogDensityFunction{Tlink},M}
    base::L
    model::M
end

function LogDensityProblems.capabilities(::Type{<:CachedPrepLDF{T,L}}) where {T,L}
    return LogDensityProblems.capabilities(L)
end
LogDensityProblems.dimension(c::CachedPrepLDF) = c.base._dim

function LogDensityProblems.logdensity(
    c::CachedPrepLDF{Tlink}, params::AbstractVector
) where {Tlink}
    b = c.base
    try
        return DynamicPPL.logdensity_at(
            params,
            c.model,
            b._getlogdensity,
            b._varname_ranges,
            b.transform_strategy,
            b._accs,
        )
    catch e
        e isa PosDefException || rethrow()
        return convert(eltype(params), -Inf)
    end
end

function LogDensityProblems.logdensity_and_gradient(
    c::CachedPrepLDF{Tlink}, params::AbstractVector
) where {Tlink}
    b = c.base
    params = convert(DynamicPPL.get_input_vector_type(b), params)
    try
        return if DynamicPPL._use_closure(b.adtype)
            DI.value_and_gradient(
                DynamicPPL.LogDensityAt{Tlink}(
                    c.model,
                    b._getlogdensity,
                    b._varname_ranges,
                    b.transform_strategy,
                    b._accs,
                ),
                b._adprep,
                b.adtype,
                params,
            )
        else
            DI.value_and_gradient(
                DynamicPPL.logdensity_at,
                b._adprep,
                b.adtype,
                params,
                DI.Constant(c.model),
                DI.Constant(b._getlogdensity),
                DI.Constant(b._varname_ranges),
                DI.Constant(b.transform_strategy),
                DI.Constant(b._accs),
            )
        end
    catch e
        e isa PosDefException || rethrow()
        T = eltype(params)
        return (convert(T, -Inf), zero(params))
    end
end

## TURING STATE ################################################################################

struct ParticleGibbsTuringState{VIT,TT,PS,LT,VNTT,PT}
    vi::VIT
    trajectory::TT
    param_state::PS
    ldf::LT
    vnt_traj::VNTT
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
    # 1. Sample all variables from prior
    vi = DynamicPPL.setacc!!(DynamicPPL.VarInfo(rng, model), TrajectoryVNTAccumulator())

    # 2. Unconditional CSMC sweep
    ctx = CSMCContext(rng, pg.csmc)
    _, vi = DynamicPPL.evaluate_nowarn!!(DynamicPPL.setleafcontext(model, ctx), vi)
    trajectory = ctx.sampled_traj[]
    vnt_traj = get_trajectory(vi)

    # 3. Condition on trajectory
    cond_model = model | vnt_traj
    θ = DynamicPPL.subset(vi, Base.filter(vn -> !(vn in keys(vnt_traj)), keys(vi)))
    ldf = DynamicPPL.LogDensityFunction(
        cond_model, DynamicPPL.getlogjoint_internal, θ; adtype=pg.adtype
    )
    cached_ldf = CachedPrepLDF(ldf, cond_model)
    _, param_state = AbstractMCMC.step(
        rng,
        AbstractMCMC.LogDensityModel(cached_ldf),
        pg.param;
        initial_params=θ[:],
        kwargs...,
    )

    # 4. Update VarInfo with new parameters
    θ_new = AbstractMCMC.getparams(cond_model, param_state)
    vi = merge(vi, DynamicPPL.unflatten!!(θ, θ_new))

    # 5. Conditional CSMC with updated parameters
    vi = DynamicPPL.setacc!!(vi, TrajectoryVNTAccumulator())
    ctx_next = CSMCContext(rng, pg.csmc; ref=trajectory)
    _, vi = DynamicPPL.evaluate_nowarn!!(DynamicPPL.setleafcontext(model, ctx_next), vi)
    trajectory_new = ctx_next.sampled_traj[]
    vnt_traj_new = get_trajectory(vi)

    # 6. Re-initialise vi in transformed space
    init_vi = merge(vi.values, vnt_traj_new)
    _, vi = DynamicPPL.init!!(
        rng, model, vi, DynamicPPL.InitFromParams(init_vi), DynamicPPL.LinkAll()
    )

    transition = DynamicPPL.ParamsWithStats(vi, model, AbstractMCMC.getstats(param_state))
    state = ParticleGibbsTuringState(vi, trajectory_new, param_state, ldf, vnt_traj_new, θ)
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

    # 2. Reuse AD prep; only swap in new conditioned model
    cached_ldf = CachedPrepLDF(state.ldf, cond_model)
    ld_model = AbstractMCMC.LogDensityModel(cached_ldf)
    # Refresh any cached log-density on `param_state` against the new conditioning.
    # Without this, MH acceptance compares the proposal's lp under the new trajectory
    # against `state.logprob` cached under the previous trajectory, biasing stationarity.
    # Mirrors Turing.jl's `setparams_varinfo!!` pattern in its Gibbs sampler.
    refreshed_param_state = AbstractMCMC.setparams!!(
        ld_model, state.param_state, AbstractMCMC.getparams(state.param_state)
    )
    _, param_state = AbstractMCMC.step(
        rng, ld_model, pg.param, refreshed_param_state; kwargs...
    )

    # 3. Update VarInfo with new parameters
    θ_new = AbstractMCMC.getparams(cond_model, param_state)
    vi = merge(vi, DynamicPPL.unflatten!!(state.θ, θ_new))

    # 4. Conditional CSMC with updated parameters
    vi = DynamicPPL.setacc!!(vi, TrajectoryVNTAccumulator())
    ctx = CSMCContext(rng, pg.csmc; ref=state.trajectory)
    _, vi = DynamicPPL.evaluate_nowarn!!(DynamicPPL.setleafcontext(model, ctx), vi)
    trajectory_new = ctx.sampled_traj[]
    vnt_traj_new = get_trajectory(vi)

    # 5. Re-initialise vi in linked space for next iteration
    init_vi = merge(vi.values, vnt_traj_new)
    _, vi = DynamicPPL.init!!(
        rng, model, vi, DynamicPPL.InitFromParams(init_vi), DynamicPPL.LinkAll()
    )

    transition = DynamicPPL.ParamsWithStats(vi, model, AbstractMCMC.getstats(param_state))
    new_state = ParticleGibbsTuringState(
        vi, trajectory_new, param_state, state.ldf, vnt_traj_new, state.θ
    )
    return transition, new_state
end
end
