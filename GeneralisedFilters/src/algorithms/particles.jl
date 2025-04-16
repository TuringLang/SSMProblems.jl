export BootstrapFilter, BF
export ParticleFilter, PF, AbstractProposal
export AuxiliaryParticleFilter, APF

import SSMProblems: distribution, simulate, logdensity


abstract type AbstractProposal end

function SSMProblems.distribution(
    model::AbstractStateSpaceModel,
    prop::AbstractProposal,
    iter::Integer,
    state,
    observation;
    kwargs...,
)
    return throw(
        MethodError(distribution, (model, prop, iter, state, observation, kwargs...))
    )
end

function SSMProblems.simulate(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    prop::AbstractProposal,
    iter::Integer,
    state,
    observation;
    kwargs...,
)
    return rand(
        rng, SSMProblems.distribution(model, prop, iter, state, observation; kwargs...)
    )
end

function SSMProblems.logdensity(
    model::AbstractStateSpaceModel,
    prop::AbstractProposal,
    iter::Integer,
    prev_state,
    new_state,
    observation;
    kwargs...,
)
    return logpdf(
        SSMProblems.distribution(model, prop, iter, prev_state, observation; kwargs...),
        new_state,
    )
end

abstract type AbstractParticleFilter <: AbstractFilter end

struct ParticleFilter{RS,PT} <: AbstractParticleFilter
    N::Int
    resampler::RS
    proposal::PT
end

const PF = ParticleFilter

function ParticleFilter(
    N::Integer, proposal::PT; threshold::Real=1.0, resampler::AbstractResampler=Systematic()
) where {PT<:AbstractProposal}
    conditional_resampler = ESSResampler(threshold, resampler)
    return ParticleFilter{ESSResampler,PT}(N, conditional_resampler, proposal)
end

function step(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    algo::AbstractParticleFilter,
    iter::Integer,
    state,
    observation;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    callback::Union{AbstractCallback,Nothing}=nothing,
    kwargs...,
)
    # capture the marginalized log-likelihood
    # TODO: Add a presampling step for the auxiliary particle filter
    update_weights!(state, model, algo, iter, observation; kwargs...) 
    state = resample(rng, algo.resampler, state; ref_state)
    reset_weights!(state, algo) # Reset weights if needed
    
    marginalization_term = logsumexp(state.log_weights)
    isnothing(callback) ||
        callback(model, algo, iter, state, observation, PostResample; kwargs...)

    state = predict(rng, model, algo, iter, state, observation; ref_state, kwargs...)
    isnothing(callback) ||
        callback(model, algo, iter, state, observation, PostPredict; kwargs...)

    # TODO: ll_increment is no longer consistent with the Kalman filter
    state, ll_increment = update(model, algo, iter, state, observation; kwargs...)
    isnothing(callback) ||
        callback(model, algo, iter, state, observation, PostUpdate; kwargs...)

    return state, (ll_increment - marginalization_term)
end

function initialise(
    rng::AbstractRNG,
    model::StateSpaceModel{T},
    filter::ParticleFilter;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
) where {T}
    particles = map(1:(filter.N)) do i
        if !isnothing(ref_state) && i == 1
            ref_state[0]
        else
            SSMProblems.simulate(rng, model.dyn; kwargs...)
        end
    end
    log_ws = zeros(T, filter.N)

    return ParticleDistribution(particles, log_ws)
end

function predict(
    rng::AbstractRNG,
    model::StateSpaceModel,
    filter::ParticleFilter,
    iter::Integer,
    state::ParticleDistribution,
    observation;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    proposed_particles = map(enumerate(state.particles)) do (i, particle)
        if !isnothing(ref_state) && i == 1
            ref_state[iter]
        else
            simulate(rng, model, filter.proposal, iter, particle, observation; kwargs...)
        end
    end

    state.log_weights +=
        map(zip(proposed_particles, state.particles)) do (new_state, prev_state)
            log_f = SSMProblems.logdensity(
                model.dyn, iter, prev_state, new_state; kwargs...
            )

            log_q = SSMProblems.logdensity(
                model, filter.proposal, iter, prev_state, new_state, observation; kwargs...
            )

            (log_f - log_q)
        end

    state.particles = proposed_particles

    return state
end

function update(
    model::StateSpaceModel{T},
    filter::ParticleFilter,
    iter::Integer,
    state::ParticleDistribution,
    observation;
    kwargs...,
) where {T}
    log_increments = map(
        x -> SSMProblems.logdensity(model.obs, iter, x, observation; kwargs...),
        state.particles,
    )

    state.log_weights += log_increments

    return state, logsumexp(state.log_weights)
end

struct LatentProposal <: AbstractProposal end

const BootstrapFilter{RS} = ParticleFilter{RS,LatentProposal}
const BF = BootstrapFilter
BootstrapFilter(N::Integer; kwargs...) = ParticleFilter(N, LatentProposal(); kwargs...)

function simulate(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    prop::LatentProposal,
    iter::Integer,
    state,
    observation;
    kwargs...,
)
    return SSMProblems.simulate(rng, model.dyn, iter, state; kwargs...)
end

function logdensity(
    model::AbstractStateSpaceModel,
    prop::LatentProposal,
    iter::Integer,
    prev_state,
    new_state,
    observation;
    kwargs...,
)
    return SSMProblems.logdensity(model.dyn, iter, prev_state, new_state; kwargs...)
end

# overwrite predict for the bootstrap filter to remove redundant computation
function predict(
    rng::AbstractRNG,
    model::StateSpaceModel,
    filter::BootstrapFilter,
    iter::Integer,
    state::ParticleDistribution,
    observation=nothing;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    state.particles = map(enumerate(state.particles)) do (i, particle)
        if !isnothing(ref_state) && i == 1
            ref_state[iter]
        else
            SSMProblems.simulate(rng, model.dyn, iter, particle; kwargs...)
        end
    end

    return state
end

# Application of particle filter to hierarchical models
function filter(
    rng::AbstractRNG,
    model::HierarchicalSSM,
    algo::ParticleFilter,
    observations::AbstractVector;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    ssm = StateSpaceModel(
        HierarchicalDynamics(model.outer_dyn, model.inner_model.dyn),
        HierarchicalObservations(model.inner_model.obs),
    )
    return filter(rng, ssm, algo, observations; ref_state=ref_state, kwargs...)
end

### AuxiliaryParticleFilter
struct AuxiliaryParticleFilter{RS,P,WT} <: AbstractParticleFilter
    N::Int
    aux::Array{WT}
    resampler::RS
    proposal::P
end

const APF = AuxiliaryParticleFilter

function AuxiliaryParticleFilter(
    N::Integer, proposal::PT; threshold::Real=1.0, resampler::AbstractResampler=Systematic()
) where {PT<:AbstractProposal}
    conditional_resampler = ESSResampler(threshold, resampler)
    return ParticleFilter{ESSResampler,PT}(N, conditional_resampler, proposal)
end

APF(N::Int; kwargs...) = AuxiliaryParticleFilter(N, LatentProposal(); kwargs...)

update_weights!(state, model, algo, iter, observation; kwargs...) = state

function update_weights!(
    state::ParticleDistribution,
    model::StateSpaceModel,
    algo::AuxiliaryParticleFilter,
    observation,
    step::Int;
    kwargs...
)
    # TODO: Can we dispatch on model capabilities maybe ?
    auxiliary_log_weights = map(enumerate(state.particles)) do (i, particle)
        logeta(particle, model, step, observation; kwargs...)
    end
    algo.aux = auxiliary_log_weights
    state.log_weights += auxiliary_log_weights
end

reset_weights!(state::ParticleDistribution, algo::AuxiliaryParticleFilter) = state.log_weights = state.log_weights[state.ancestors] - algo.aux[state.ancestors]
reset_weights!(state::ParticleDistribution, algo::ParticleFilter) = state # Wonky, construction of ParticleDistribution
