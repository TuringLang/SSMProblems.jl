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
    vi::DynamicPPL.AbstractVarInfo,
)
    return DynamicPPL.tilde_assume!!(DynamicPPL.DefaultContext(), dist, vn, vi)
end

function DynamicPPL.tilde_observe!!(
    ::CSMCContext,
    right::Distributions.Distribution,
    left,
    vn::Union{DynamicPPL.VarName,Nothing},
    vi::DynamicPPL.AbstractVarInfo,
)
    return DynamicPPL.tilde_observe!!(DynamicPPL.DefaultContext(), right, left, vn, vi)
end

## TURING STATE ################################################################################

struct ParticleGibbsTuringState{VIT,TT,VNT,PS,CVI}
    vi::VIT
    trajectory::TT
    traj_vn::VNT
    param_state::PS
    cond_vi_linked::CVI
end

## HELPERS #####################################################################################

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
    vi_updated = DynamicPPL.unflatten(cond_vi_linked, θ_new)
    vi_constrained = DynamicPPL.invlink(vi_updated, cond_model)
    return DynamicPPL.values_as(vi_constrained, NamedTuple)
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
    vi = DynamicPPL.setindex!!(vi, traj_flat, traj_vn)
    vi = last(DynamicPPL.evaluate!!(model, vi))

    # 5. Condition on trajectory, create LogDensityFunction for parameter step
    cond_model = _condition_on_trajectory(model, traj_vn, traj_flat)
    ldf, cond_vi_linked = _make_conditioned_ldf(cond_model, vi, traj_vn, pg.adtype)
    ld_model = AbstractMCMC.LogDensityModel(ldf)

    # 6. Initial parameter step
    θ = cond_vi_linked[:]
    _, param_state = AbstractMCMC.step(rng, ld_model, pg.param; initial_params=θ, kwargs...)

    # 7. Update VarInfo with new parameters, discover new SSM
    θ_new = AbstractMCMC.getparams(param_state)
    param_vals = _recover_params(cond_vi_linked, cond_model, θ_new)
    for (k, v) in pairs(param_vals)
        vn = DynamicPPL.VarName{k}()
        vi = DynamicPPL.setindex!!(vi, v, vn)
    end
    vi = last(DynamicPPL.evaluate!!(model, vi))

    # 8. Discover new SSM and run CSMC
    ssm_dist_new, _ = _discover_ssm(model, vi)
    trajectory_new, _ = _csmc_sample(
        rng, ssm_dist_new.model, pg.csmc, ssm_dist_new.observations, trajectory
    )

    # 9. Update trajectory in VarInfo
    outer_traj_new = _outer_trajectory(trajectory_new, af)
    traj_flat_new = _flatten_trajectory(outer_traj_new, T_len, Dx)
    vi = DynamicPPL.setindex!!(vi, traj_flat_new, traj_vn)
    vi = last(DynamicPPL.evaluate!!(model, vi))

    transition = ParticleGibbsTransition(θ_new, AbstractMCMC.getstats(param_state))
    state = ParticleGibbsTuringState(
        vi, trajectory_new, traj_vn, param_state, cond_vi_linked
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

    # 3. Create LogDensityFunction (new each step since trajectory changed)
    ldf, cond_vi_linked = _make_conditioned_ldf(cond_model, vi, traj_vn, pg.adtype)
    ld_model = AbstractMCMC.LogDensityModel(ldf)

    # 4. Parameter step (preserves adaptation via state.param_state)
    _, param_state = AbstractMCMC.step(
        rng, ld_model, pg.param, state.param_state; kwargs...
    )

    # 5. Extract new θ and update VarInfo
    θ_new = AbstractMCMC.getparams(param_state)
    param_vals = _recover_params(cond_vi_linked, cond_model, θ_new)
    for (k, v) in pairs(param_vals)
        vn = DynamicPPL.VarName{k}()
        vi = DynamicPPL.setindex!!(vi, v, vn)
    end
    vi = last(DynamicPPL.evaluate!!(model, vi))

    # 6. Discover new SSM and run CSMC
    ssm_dist_new, _ = _discover_ssm(model, vi)
    trajectory_new, _ = _csmc_sample(
        rng, ssm_dist_new.model, pg.csmc, ssm_dist_new.observations, state.trajectory
    )

    # 7. Update trajectory in VarInfo
    outer_traj_new = _outer_trajectory(trajectory_new, af)
    traj_flat_new = _flatten_trajectory(outer_traj_new, T_len, Dx)
    vi = DynamicPPL.setindex!!(vi, traj_flat_new, traj_vn)
    vi = last(DynamicPPL.evaluate!!(model, vi))

    transition = ParticleGibbsTransition(θ_new, AbstractMCMC.getstats(param_state))
    new_state = ParticleGibbsTuringState(
        vi, trajectory_new, traj_vn, param_state, cond_vi_linked
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
    nt = DynamicPPL.values_as(vi, NamedTuple)
    names = Symbol[]
    for (k, v) in pairs(nt)
        if v isa AbstractArray && length(v) > 1
            for i in 1:length(v)
                push!(names, Symbol("$(k)[$i]"))
            end
        else
            push!(names, k)
        end
    end
    return names
end
