export BootstrapFilter, BF

struct BootstrapFilter{RS<:AbstractConditionalResampler} <: AbstractFilter
    N::Integer
    resampler::RS
end

function BF(N::Integer; threshold::Real=1.0, resampler::AbstractResampler=Systematic())
    conditional_resampler = ESSResampler(threshold, resampler)
    return BootstrapFilter(N, conditional_resampler)
end

function initialise(
    rng::AbstractRNG,
    model::StateSpaceModel,
    filter::BootstrapFilter;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    initial_states = map(x -> SSMProblems.simulate(rng, model.dyn; kwargs...), 1:(filter.N))
    initial_weights = zeros(eltype(model), filter.N)

    return update_ref!(ParticleContainer(initial_states, initial_weights), ref_state)
end

function predict(
    rng::AbstractRNG,
    model::StateSpaceModel,
    filter::BootstrapFilter,
    step::Integer,
    states::ParticleContainer{T};
    ref_state::Union{Nothing,AbstractVector{T}}=nothing,
    kwargs...,
) where {T}
    states.ancestors = resample(rng, filter.resampler, states.filtered)
    states.proposed.particles = map(
        x -> SSMProblems.simulate(rng, model.dyn, step, x; kwargs...),
        states.filtered[states.ancestors],
    )

    states.proposed.log_weights = states.filtered.log_weights
    return update_ref!(states, ref_state, step)
end

function update(
    model::StateSpaceModel,
    filter::BootstrapFilter,
    step::Integer,
    states::ParticleContainer,
    observation;
    kwargs...,
)
    log_marginals = map(
        x -> SSMProblems.logdensity(model.obs, step, x, observation; kwargs...),
        collect(states.proposed),
    )

    states.filtered.log_weights += log_marginals
    states.filtered.particles = states.proposed.particles

    log_marginal = logmarginal(states)
    return (states, log_marginal)
end
