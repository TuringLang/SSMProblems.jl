export GuidedFilter, GPF, AbstractProposal
# import SSMProblems: distribution, simulate, logdensity

"""
    AbstractProposal
"""
abstract type AbstractProposal end

# TODO: improve this and ensure that there are no conflicts with SSMProblems
function distribution(
    model::AbstractStateSpaceModel,
    prop::AbstractProposal,
    step::Integer,
    state,
    observation;
    kwargs...,
)
    return throw(
        MethodError(distribution, (model, prop, step, state, observation, kwargs...))
    )
end

function simulate(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    prop::AbstractProposal,
    step::Integer,
    state,
    observation;
    kwargs...,
)
    return rand(rng, distribution(model, prop, step, state, observation; kwargs...))
end

function logdensity(
    model::AbstractStateSpaceModel,
    prop::AbstractProposal,
    step::Integer,
    prev_state,
    new_state,
    observation;
    kwargs...,
)
    return logpdf(
        distribution(model, prop, step, prev_state, observation; kwargs...), new_state
    )
end

struct GuidedFilter{RS<:AbstractResampler,P<:AbstractProposal} <: AbstractParticleFilter
    N::Integer
    resampler::RS
    proposal::P
end

function GuidedFilter(
    N::Integer, proposal::PT; threshold::Real=1.0, resampler::AbstractResampler=Systematic()
) where {PT<:AbstractProposal}
    conditional_resampler = ESSResampler(threshold, resampler)
    return GuidedFilter{ESSResampler,PT}(N, conditional_resampler, proposal)
end

"""Shorthand for `GuidedFilter`"""
const GPF = GuidedFilter

function initialise(
    rng::AbstractRNG,
    model::StateSpaceModel{T},
    filter::GuidedFilter;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
) where {T}
    particles = map(x -> SSMProblems.simulate(rng, model.dyn; kwargs...), 1:(filter.N))
    weights = zeros(T, filter.N)

    return update_ref!(ParticleDistribution(particles, weights), ref_state)
end

function predict(
    rng::AbstractRNG,
    model::StateSpaceModel,
    filter::GuidedFilter,
    step::Integer,
    state::ParticleDistribution,
    observation;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    proposed_particles = map(
        x -> simulate(rng, model, filter.proposal, step, x, observation; kwargs...),
        collect(state),
    )

    log_increments =
        map(zip(proposed_particles, state.particles)) do (new_state, prev_state)
            log_f = SSMProblems.logdensity(
                model.dyn, step, prev_state, new_state; kwargs...
            )
            log_q = logdensity(
                model, filter.proposal, step, prev_state, new_state, observation; kwargs...
            )

            (log_f - log_q)
        end

    proposed_state = ParticleDistribution(
        proposed_particles, state.log_weights + log_increments
    )

    return update_ref!(proposed_state, ref_state, step)
end

function update(
    model::StateSpaceModel{T},
    filter::GuidedFilter,
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
