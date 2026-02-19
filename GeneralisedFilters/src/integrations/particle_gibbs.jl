using LogDensityProblemsAD: LogDensityProblemsAD
using MCMCChains: MCMCChains

export ParticleGibbs, ParticleGibbsModel

## TYPES #####################################################################################

"""
    ParticleGibbs{CS, PS, ADT} <: AbstractMCMC.AbstractSampler

Particle Gibbs sampler that alternates between a parameter update (e.g., NUTS) and a
trajectory update (conditional SMC).

# Fields
- `csmc::CS`: Conditional SMC sampler for trajectory updates (e.g., `CSMCAS(RBPF(BF(200), KF()))`)
- `param::PS`: Parameter sampler (e.g., `AdvancedHMC.NUTS(0.8)`)
- `adtype::ADT`: AD backend (`ADTypes.AbstractADType`). `nothing` uses AdvancedHMC's default
  (ForwardDiff). For HierarchicalSSM models, specify a reverse-mode backend that uses
  ChainRules (e.g., `AutoZygote()`). Requires the corresponding package to be loaded.

# Examples
```julia

# Regular SSM
ParticleGibbs(CSMC(BF(100)), NUTS(0.8))

# Hierarchical SSM (needs reverse-mode AD for KF rrule)
ParticleGibbs(CSMCAS(RBPF(BF(200), KF())), NUTS(0.8); adtype=AutoZygote())
```
"""
struct ParticleGibbs{CS<:ConditionalSMC,PS,ADT<:Union{Nothing,ADTypes.AbstractADType}} <:
       AbstractMCMC.AbstractSampler
    csmc::CS
    param::PS
    adtype::ADT
end

function ParticleGibbs(csmc::ConditionalSMC, param; adtype=nothing)
    return ParticleGibbs(csmc, param, adtype)
end

"""
    ParticleGibbsModel{PT, MT} <: AbstractMCMC.AbstractModel

Model for particle Gibbs inference, combining a prior on parameters with a parameterised SSM.

# Fields
- `prior::PT`: Prior distribution on θ (any Distributions.jl distribution)
- `param_model::MT`: A `ParameterisedSSM` mapping θ to a concrete SSM

# Examples
```julia
pssm = ParameterisedSSM(θ -> build_model(θ, fixed), observations)
model = ParticleGibbsModel(MvNormal(zeros(d), 4.0*I), pssm)
```
"""
struct ParticleGibbsModel{PT,MT<:ParameterisedSSM} <: AbstractMCMC.AbstractModel
    prior::PT
    param_model::MT
end

"""
    ParticleGibbsState{VT, TT, PS, LDT}

Internal state of the particle Gibbs sampler.

# Fields
- `θ`: Current parameter vector
- `trajectory`: Current reference trajectory (OffsetVector)
- `param_state`: Parameter sampler state (e.g., `AdvancedHMC.HMCState`)
- `log_density`: `AbstractMCMC.LogDensityModel` wrapping the SSM log-density (persisted
  so the trajectory `Ref` can be updated between steps)
"""
struct ParticleGibbsState{VT,TT,PS,LDT}
    θ::VT
    trajectory::TT
    param_state::PS
    log_density::LDT
end

"""
    ParticleGibbsTransition{VT, NT}

A single transition of the particle Gibbs sampler, containing the parameter values and
diagnostics from the parameter sampler.
"""
struct ParticleGibbsTransition{VT,NT<:NamedTuple}
    θ::VT
    stat::NT
end

## INNER FILTER EXTRACTION ####################################################################

_get_inner_filter(::AbstractParticleFilter) = nothing
_get_inner_filter(pf::RBPF) = pf.af

# Extract outer trajectory for the log-density (which only needs x, not the inner distribution)
_outer_trajectory(trajectory, ::Nothing) = trajectory
_outer_trajectory(trajectory, ::AbstractFilter) = map(s -> s.x, trajectory)

## LOG-DENSITY MODEL CONSTRUCTION #############################################################

function _get_traj_ref(ld_model::AbstractMCMC.LogDensityModel)
    ld = ld_model.logdensity
    # Unwrap LogDensityProblemsAD wrapper if present
    inner = if hasproperty(ld, :ℓ)
        ld.ℓ
    else
        ld
    end
    return inner.trajectory
end

function _create_log_density_model(
    model::ParticleGibbsModel, af, trajectory, adtype::Nothing
)
    if !isnothing(af)
        throw(
            ArgumentError(
                "HierarchicalSSM models require a reverse-mode AD backend for gradient " *
                "computation (the ChainRules rrule on kf_loglikelihood is not picked up " *
                "by ForwardDiff). Specify `adtype=AutoZygote()` (or another reverse-mode " *
                "backend) when constructing ParticleGibbs, and load the corresponding " *
                "package (e.g., `using Zygote`).",
            ),
        )
    end
    ld = SSMParameterLogDensity(model.prior, model.param_model, Ref(trajectory))
    return AbstractMCMC.LogDensityModel(ld)
end

function _create_log_density_model(
    model::ParticleGibbsModel, af, trajectory, adtype::ADTypes.AbstractADType
)
    ld = if isnothing(af)
        SSMParameterLogDensity(model.prior, model.param_model, Ref(trajectory))
    else
        SSMParameterLogDensity(model.prior, model.param_model, af, Ref(trajectory))
    end
    ld_with_grad = LogDensityProblemsAD.ADgradient(adtype, ld)
    return AbstractMCMC.LogDensityModel(ld_with_grad)
end

## ABSTRACTMCMC INTERFACE #####################################################################

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::ParticleGibbsModel,
    pg::ParticleGibbs;
    initial_params=nothing,
    kwargs...,
)
    # Sample initial θ
    θ = if isnothing(initial_params)
        Vector{Float64}(rand(rng, model.prior))
    else
        Vector{Float64}(initial_params)
    end

    # Build SSM and run unconditional CSMC for initial trajectory
    ssm = model.param_model.build(θ)
    af = _get_inner_filter(pg.csmc.pf)
    trajectory, _ = _csmc_sample(rng, ssm, pg.csmc, model.param_model.observations, nothing)

    # Create log-density model (uses outer-only trajectory for hierarchical models)
    outer_traj = _outer_trajectory(trajectory, af)
    ld_model = _create_log_density_model(model, af, outer_traj, pg.adtype)

    # Run initial parameter step
    _, param_state = AbstractMCMC.step(rng, ld_model, pg.param; initial_params=θ, kwargs...)

    # Extract new θ and run CSMC
    θ_new = AbstractMCMC.getparams(param_state)
    ssm_new = model.param_model.build(θ_new)
    trajectory_new, _ = _csmc_sample(
        rng, ssm_new, pg.csmc, model.param_model.observations, trajectory
    )

    # Update trajectory ref with outer-only trajectory
    traj_ref = _get_traj_ref(ld_model)
    traj_ref[] = _outer_trajectory(trajectory_new, af)

    transition = ParticleGibbsTransition(θ_new, AbstractMCMC.getstats(param_state))
    state = ParticleGibbsState(θ_new, trajectory_new, param_state, ld_model)

    return transition, state
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::ParticleGibbsModel,
    pg::ParticleGibbs,
    state::ParticleGibbsState;
    kwargs...,
)
    # Update trajectory ref so the log-density reflects the current trajectory
    af = _get_inner_filter(pg.csmc.pf)
    traj_ref = _get_traj_ref(state.log_density)
    traj_ref[] = _outer_trajectory(state.trajectory, af)

    # Run parameter step (picks up updated trajectory Ref)
    _, param_state = AbstractMCMC.step(
        rng, state.log_density, pg.param, state.param_state; kwargs...
    )

    # Extract new θ and run CSMC (pass full trajectory for conditioning)
    θ_new = AbstractMCMC.getparams(param_state)
    ssm_new = model.param_model.build(θ_new)
    trajectory_new, _ = _csmc_sample(
        rng, ssm_new, pg.csmc, model.param_model.observations, state.trajectory
    )

    transition = ParticleGibbsTransition(θ_new, AbstractMCMC.getstats(param_state))
    new_state = ParticleGibbsState(θ_new, trajectory_new, param_state, state.log_density)

    return transition, new_state
end

## CHAIN OUTPUT ###############################################################################

function AbstractMCMC.bundle_samples(
    ts::Vector{<:ParticleGibbsTransition},
    ::ParticleGibbsModel,
    ::ParticleGibbs,
    state,
    ::Type{MCMCChains.Chains};
    param_names=nothing,
    kwargs...,
)
    n_samples = length(ts)
    d = length(first(ts).θ)

    # Parameter names
    names = if isnothing(param_names)
        [Symbol("θ[$i]") for i in 1:d]
    else
        Symbol.(param_names)
    end

    # Build parameter matrix
    vals = Matrix{Float64}(undef, n_samples, d)
    for (i, t) in enumerate(ts)
        vals[i, :] = t.θ
    end

    # Internal stats (derived from the NamedTuple keys in each transition)
    int_names = collect(Symbol, keys(first(ts).stat))
    internals = Matrix{Float64}(undef, n_samples, length(int_names))
    for (i, t) in enumerate(ts)
        for (j, v) in enumerate(values(t.stat))
            internals[i, j] = Float64(v)
        end
    end

    all_vals = hcat(vals, internals)
    all_names = vcat(names, int_names)
    return MCMCChains.Chains(all_vals, all_names, (parameters=names, internals=int_names))
end
