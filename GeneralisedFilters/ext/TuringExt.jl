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
using LogDensityProblems: LogDensityProblems
using MCMCChains: MCMCChains
using OffsetArrays
using Random: AbstractRNG
using SSMProblems
using Turing: Turing

## BIJECTORS INTEGRATION #######################################################################

bijector(::SSMTrajectory) = identity

Bijectors.VectorBijectors.to_linked_vec(::SSMTrajectory) = TypedIdentity()
Bijectors.VectorBijectors.from_linked_vec(::SSMTrajectory) = TypedIdentity()
Bijectors.VectorBijectors.linked_vec_length(d::SSMTrajectory) = length(d)

## CSMC CONTEXT ################################################################################

"""
    CSMCContext <: DynamicPPL.AbstractContext

A DynamicPPL leaf context that intercepts `x ~ SSMTrajectory(...)` during model evaluation,
capturing the distribution and variable name so the sampler can extract the SSM and run CSMC.

For non-SSMTrajectory variables, delegates to `DefaultContext`.
"""
struct CSMCContext <: DynamicPPL.AbstractContext
    ssm_dist::Ref{Any}
    traj_vn::Ref{Any}
end

CSMCContext() = CSMCContext(Ref{Any}(nothing), Ref{Any}(nothing))

function DynamicPPL.tilde_assume!!(
    ctx::CSMCContext,
    dist::SSMTrajectory,
    vn::DynamicPPL.VarName,
    ::Any,
    vi::DynamicPPL.AbstractVarInfo,
)
    ctx.ssm_dist[] = dist
    ctx.traj_vn[] = vn
    x = DynamicPPL.getindex_internal(vi, vn)
    return x, vi
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
    return DynamicPPL.logdensity_at(
        params, c.model, b._getlogdensity, b._varname_ranges, b.transform_strategy, b._accs
    )
end

function LogDensityProblems.logdensity_and_gradient(
    c::CachedPrepLDF{Tlink}, params::AbstractVector
) where {Tlink}
    b = c.base
    params = convert(DynamicPPL.get_input_vector_type(b), params)
    return if DynamicPPL._use_closure(b.adtype)
        DI.value_and_gradient(
            DynamicPPL.LogDensityAt{Tlink}(
                c.model, b._getlogdensity, b._varname_ranges, b.transform_strategy, b._accs
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
end

## TURING STATE ################################################################################

struct ParticleGibbsTuringState{VIT,TT,VNT,PS,CVI,LT}
    vi::VIT
    trajectory::TT
    traj_vn::VNT
    param_state::PS
    cond_vi_linked::CVI
    ldf::LT
end

## HELPERS #####################################################################################

function get_trajectories(vi::DynamicPPL.AbstractVarInfo)
    return DynamicPPL.getacc(vi, Val(SSM_ACCUMULATOR)).values
end

function _discover_ssm(model::DynamicPPL.Model, vi::DynamicPPL.AbstractVarInfo)
    ctx = CSMCContext()
    discovery_model = DynamicPPL.setleafcontext(model, ctx)
    DynamicPPL.evaluate!!(discovery_model, vi)
    return ctx.ssm_dist[]::SSMTrajectory, ctx.traj_vn[]::DynamicPPL.VarName
end

function _condition_on_trajectory(
    model::DynamicPPL.Model, traj_vn::DynamicPPL.VarName, traj_flat::AbstractVector
)
    return DynamicPPL.condition(model, traj_vn => traj_flat)
end

"""
    _make_conditioned_ldf(cond_model, vi, traj_vn, adtype)

Create a `LogDensityFunction` for the conditioned model (trajectory fixed) plus
a linked VarInfo for parameter recovery after NUTS steps.

Returns `(ldf, cond_vi_linked)`.
"""
function _make_conditioned_ldf(
    cond_model::DynamicPPL.Model,
    vi::DynamicPPL.AbstractVarInfo,
    traj_vn::DynamicPPL.VarName,
    adtype,
)
    param_vns = Base.filter(vn -> vn != traj_vn, keys(vi))
    cond_vi = DynamicPPL.subset(vi, param_vns)
    cond_vi_linked = DynamicPPL.link(cond_vi, cond_model)
    ldf = DynamicPPL.LogDensityFunction(
        cond_model, DynamicPPL.getlogjoint_internal, cond_vi_linked; adtype=adtype
    )
    return ldf, cond_vi_linked
end

"""
    _recover_params(cond_vi_linked, cond_model, θ_new)

Given a linked VarInfo template and new unconstrained parameters from NUTS,
recover the constrained parameter values as a NamedTuple.
"""
function _recover_params(cond_vi_linked, cond_model, θ_new)
    vi_updated = DynamicPPL.unflatten!!(cond_vi_linked, θ_new)
    vi_constrained = DynamicPPL.invlink(vi_updated, cond_model)
    return vi_constrained
end

## ABSTRACTMCMC INTERFACE ######################################################################

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    pg::ParticleGibbs;
    initial_params=nothing,
    kwargs...,
)
    # 1. Create VarInfo (samples all variables from prior)
    vi = DynamicPPL.VarInfo(rng, model)

    # 2. Discover trajectory variable
    ssm_dist, traj_vn = _discover_ssm(model, vi)

    # 3. Run unconditional CSMC for initial trajectory
    af = _get_inner_filter(pg.csmc.pf)
    trajectory, _ = _csmc_sample(
        rng, ssm_dist.model, pg.csmc, ssm_dist.observations, nothing
    )

    # 4. Flatten trajectory and update VarInfo
    T_len = length(ssm_dist.observations)
    Dx = _state_dim(ssm_dist)
    outer_traj = _outer_trajectory(trajectory, af)
    traj_flat = _flatten_trajectory(outer_traj, T_len, Dx)
    vi = DynamicPPL.setindex_internal!!(vi, traj_flat, traj_vn)
    vi = last(DynamicPPL.evaluate!!(model, vi))

    # 5. Condition on trajectory, create LogDensityFunction for parameter step
    cond_model = _condition_on_trajectory(model, traj_vn, traj_flat)
    ldf, cond_vi_linked = _make_conditioned_ldf(cond_model, vi, traj_vn, pg.adtype)
    ld_model = AbstractMCMC.LogDensityModel(ldf)

    # 6. Initial parameter step
    θ = cond_vi_linked[:]
    _, param_state = AbstractMCMC.step(rng, ld_model, pg.param; initial_params=θ, kwargs...)

    # 7. Update VarInfo with new parameters, discover new SSM
    θ_new = AbstractMCMC.getparams(cond_model, param_state)
    param_vals = _recover_params(cond_vi_linked, cond_model, θ_new)
    vi = merge(vi, param_vals)
    vi = last(DynamicPPL.evaluate!!(model, vi))

    # 8. Discover new SSM and run CSMC
    ssm_dist_new, _ = _discover_ssm(model, vi)
    trajectory_new, _ = _csmc_sample(
        rng, ssm_dist_new.model, pg.csmc, ssm_dist_new.observations, trajectory
    )

    # 9. Update trajectory in VarInfo
    outer_traj_new = _outer_trajectory(trajectory_new, af)
    traj_flat_new = _flatten_trajectory(outer_traj_new, T_len, Dx)
    init_vi = DynamicPPL.setindex!!(vi.values, traj_flat_new, traj_vn)
    init_strategy = DynamicPPL.InitFromParams(init_vi)
    _, vi = DynamicPPL.init!!(rng, model, vi, init_strategy, DynamicPPL.LinkAll())

    transition = ParticleGibbsTransition(θ_new, AbstractMCMC.getstats(param_state))
    state = ParticleGibbsTuringState(
        vi, trajectory_new, traj_vn, param_state, cond_vi_linked, ldf
    )

    return transition, state
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    pg::ParticleGibbs,
    state::ParticleGibbsTuringState;
    kwargs...,
)
    af = _get_inner_filter(pg.csmc.pf)
    vi = state.vi
    traj_vn = state.traj_vn

    # 1. Get trajectory dimensions
    ssm_dist, _ = _discover_ssm(model, vi)
    T_len = length(ssm_dist.observations)
    Dx = _state_dim(ssm_dist)

    # 2. Condition on current trajectory
    outer_traj = _outer_trajectory(state.trajectory, af)
    traj_flat = _flatten_trajectory(outer_traj, T_len, Dx)
    cond_model = _condition_on_trajectory(model, traj_vn, traj_flat)

    # 3. Reuse AD prep from initial step; only swap in new conditioned model
    cond_vi_linked = state.cond_vi_linked
    cached_ldf = CachedPrepLDF(state.ldf, cond_model)
    ld_model = AbstractMCMC.LogDensityModel(cached_ldf)

    # 4. Parameter step (preserves adaptation via state.param_state)
    _, param_state = AbstractMCMC.step(
        rng, ld_model, pg.param, state.param_state; kwargs...
    )

    # 5. Extract new θ and update VarInfo
    θ_new = AbstractMCMC.getparams(cond_model, param_state)
    param_vals = _recover_params(cond_vi_linked, cond_model, θ_new)
    vi = merge(vi, param_vals)
    vi = last(DynamicPPL.evaluate!!(model, vi))

    # 6. Discover new SSM and run CSMC
    ssm_dist_new, _ = _discover_ssm(model, vi)
    trajectory_new, _ = _csmc_sample(
        rng, ssm_dist_new.model, pg.csmc, ssm_dist_new.observations, state.trajectory
    )

    # 7. Update trajectory in VarInfo
    outer_traj_new = _outer_trajectory(trajectory_new, af)
    traj_flat_new = _flatten_trajectory(outer_traj_new, T_len, Dx)
    init_vi = DynamicPPL.setindex!!(vi.values, traj_flat_new, traj_vn)
    init_strategy = DynamicPPL.InitFromParams(init_vi)
    _, vi = DynamicPPL.init!!(rng, model, vi, init_strategy, DynamicPPL.LinkAll())

    transition = ParticleGibbsTransition(θ_new, AbstractMCMC.getstats(param_state))
    new_state = ParticleGibbsTuringState(
        vi, trajectory_new, traj_vn, param_state, cond_vi_linked, state.ldf
    )

    return transition, new_state
end

## CHAIN OUTPUT ################################################################################

function AbstractMCMC.bundle_samples(
    ts::Vector{<:ParticleGibbsTransition},
    ::DynamicPPL.Model,
    ::ParticleGibbs,
    state::ParticleGibbsTuringState,
    ::Type{MCMCChains.Chains};
    param_names=nothing,
    kwargs...,
)
    names = if isnothing(param_names)
        _turing_param_names(state)
    else
        Symbol.(param_names)
    end
    return _build_chains(ts, names)
end

function _turing_param_names(state::ParticleGibbsTuringState)
    vi = state.cond_vi_linked
    # nt = DynamicPPL.values_as(vi, NamedTuple)
    vnt = DynamicPPL.get_values(vi)
    names = Symbol[]
    for (k, v) in pairs(vnt)
        if v isa AbstractArray && length(v) > 1
            for i in 1:length(v)
                push!(names, Symbol("$(k)[$i]"))
            end
        else
            push!(names, Symbol(k))
        end
    end
    return names
end

end
