struct BootstrapFilter{T<:Real,RS<:AbstractResampler} <: AbstractFilter
    N::Int
    threshold::T
    resampler::RS
end

function BF(N::Integer; threshold::Real=1.0, resampler::AbstractResampler=Systematic())
    return BootstrapFilter(N, threshold, resampler)
end

resample_threshold(filter::BootstrapFilter) = filter.threshold * filter.N

function initialise(
    rng::AbstractRNG,
    model::StateSpaceModel,
    filter::BootstrapFilter,
    extra;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    initial_states = map(x -> SSMProblems.simulate(rng, model.dyn, extra), 1:(filter.N))
    initial_weights = zeros(eltype(model), filter.N)

    return update_ref!(ParticleContainer(initial_states, initial_weights), ref_state)
end

function resample(rng::AbstractRNG, states::ParticleContainer, filter::BootstrapFilter)
    weights = StatsBase.weights(states)
    ess = inv(sum(abs2, weights))
    @debug "ESS: $ess"

    if resample_threshold(filter) ≥ ess
        idx = resample(rng, filter.resampler, weights)
        reset_weights!(states)
    else
        idx = 1:(filter.N)
    end

    return idx
end

function predict(
    rng::AbstractRNG,
    model::StateSpaceModel,
    filter::BootstrapFilter,
    step::Integer,
    states::ParticleContainer{T},
    extra;
    ref_state::Union{Nothing,AbstractVector{T}}=nothing,
    kwargs...,
) where {T}
    states.ancestors = resample(rng, states, filter)
    states.proposed = map(
        x -> SSMProblems.simulate(rng, model.dyn, step, x, extra),
        states.filtered[states.ancestors],
    )

    return update_ref!(states, ref_state, step)
end

function update(
    model::StateSpaceModel,
    filter::BootstrapFilter,
    step::Integer,
    states::ParticleContainer,
    observation,
    extra;
    kwargs...,
)
    log_marginals = map(
        x -> SSMProblems.logdensity(model.obs, step, x, observation, extra), states.proposed
    )

    prev_log_marginal = logsumexp(states.log_weights)
    states.log_weights += log_marginals
    states.filtered = states.proposed

    return (states, logsumexp(states.log_weights) - prev_log_marginal)
end
