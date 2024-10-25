export BootstrapFilter, BF

struct BootstrapFilter{RS<:AbstractResampler} <: AbstractFilter
    N::Integer
    resampler::RS
end

"""Shorthand for `BootstrapFilter`"""
const BF = BootstrapFilter

function BootstrapFilter(
    N::Integer; threshold::Real=1.0, resampler::AbstractResampler=Systematic()
)
    conditional_resampler = ESSResampler(threshold, resampler)
    return BootstrapFilter(N, conditional_resampler)
end

function initialise(
    rng::AbstractRNG,
    model::StateSpaceModel{T},
    filter::BootstrapFilter;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
) where {T}
    initial_states = map(x -> SSMProblems.simulate(rng, model.dyn; kwargs...), 1:(filter.N))
    initial_weights = zeros(T, filter.N)

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
    states.proposed, states.ancestors = resample(rng, filter.resampler, states.filtered)
    states.proposed.particles = map(
        x -> SSMProblems.simulate(rng, model.dyn, step, x; kwargs...),
        collect(states.proposed),
    )

    return update_ref!(states, ref_state, step)
end

function update(
    model::StateSpaceModel{T},
    filter::BootstrapFilter,
    step::Integer,
    states::ParticleContainer,
    observation;
    kwargs...,
) where {T}
    log_increments = map(
        x -> SSMProblems.logdensity(model.obs, step, x, observation; kwargs...),
        collect(states.proposed),
    )

    states.filtered.log_weights = states.proposed.log_weights + log_increments
    states.filtered.particles = states.proposed.particles

    return states, logmarginal(states)
end
