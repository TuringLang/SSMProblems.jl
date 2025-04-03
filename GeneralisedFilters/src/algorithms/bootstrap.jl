export BootstrapFilter, BF

abstract type AbstractParticleFilter <: AbstractFilter end

function step(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    alg::AbstractParticleFilter,
    iter::Integer,
    state,
    observation;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    callback::Union{AbstractCallback,Nothing}=nothing,
    kwargs...,
)
    # capture the marginalized log-likelihood
    state = resample(rng, alg.resampler, state; ref_state)
    marginalization_term = logsumexp(state.log_weights)
    isnothing(callback) ||
        callback(model, alg, iter, state, observation, PostResample; kwargs...)

    state = predict(rng, model, alg, iter, state, observation; ref_state, kwargs...)
    isnothing(callback) ||
        callback(model, alg, iter, state, observation, PostPredict; kwargs...)

    state, ll_increment = update(model, alg, iter, state, observation; kwargs...)
    isnothing(callback) ||
        callback(model, alg, iter, state, observation, PostUpdate; kwargs...)

    return state, (ll_increment - marginalization_term)
end

struct BootstrapFilter{RS<:AbstractResampler} <: AbstractParticleFilter
    N::Int
    resampler::RS
end

"""Shorthand for `BootstrapFilter`"""
const BF = BootstrapFilter

function BootstrapFilter(
    N::Integer; threshold::Real=1.0, resampler::AbstractResampler=Systematic()
)
    conditional_resampler = ESSResampler(threshold, resampler)
    return BootstrapFilter{ESSResampler}(N, conditional_resampler)
end

function initialise(
    rng::AbstractRNG,
    model::StateSpaceModel{T},
    filter::BootstrapFilter;
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
    filter::BootstrapFilter,
    step::Integer,
    state::ParticleDistribution,
    observation;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    state.particles = map(enumerate(state.particles)) do (i, particle)
        if !isnothing(ref_state) && i == 1
            ref_state[step]
        else
            SSMProblems.simulate(rng, model.dyn, step, particle; kwargs...)
        end
    end

    return state
end

function update(
    model::StateSpaceModel{T},
    filter::BootstrapFilter,
    step::Integer,
    state::ParticleDistribution,
    observation;
    kwargs...,
) where {T}
    log_increments = map(
        x -> SSMProblems.logdensity(model.obs, step, x, observation; kwargs...),
        collect(state),
    )

    state.log_weights += log_increments

    return state, logsumexp(state.log_weights)
end

# Application of bootstrap filter to hierarchical models
function filter(
    rng::AbstractRNG,
    model::HierarchicalSSM,
    alg::BootstrapFilter,
    observations::AbstractVector;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    ssm = StateSpaceModel(
        HierarchicalDynamics(model.outer_dyn, model.inner_model.dyn),
        HierarchicalObservations(model.inner_model.obs),
    )
    return filter(rng, ssm, alg, observations; ref_state=ref_state, kwargs...)
end
