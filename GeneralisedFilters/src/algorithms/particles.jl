export BootstrapFilter, BF
export ParticleFilter, PF, AbstractProposal

# import SSMProblems: distribution, simulate, logdensity

abstract type AbstractProposal end

function distribution(
    model::AbstractStateSpaceModel,
    prop::AbstractProposal,
    step::Integer,
    state,
    observation;
    kwargs...,
)
    return throw(
        MethodError(
            distribution, (model, prop, step, state, observation, kwargs...)
        ),
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
    return rand(
        rng, distribution(model, prop, step, state, observation; kwargs...)
    )
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
        distribution(model, prop, step, prev_state, observation; kwargs...),
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
    alg::AbstractParticleFilter,
    iter::Integer,
    state,
    observation;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    callback::Union{AbstractCallback,Nothing}=nothing,
    kwargs...,
)
    # capture the marginalized log-likelihood
    state = resample(rng, alg.resampler, state)
    marginalization_term = logsumexp(state.log_weights)
    isnothing(callback) ||
        callback(model, alg, iter, state, observation, PostResample; kwargs...)

    state = predict(
        rng, model, alg, iter, state, observation; ref_state=ref_state, kwargs...
    )

    # TODO: this is quite inelegant and should be refactored. It also might introduce bugs
    # with callbacks that track the ancestry (and use PostResample)
    if !isnothing(ref_state)
        CUDA.@allowscalar state.ancestors[1] = 1
    end
    isnothing(callback) ||
        callback(model, alg, iter, state, observation, PostPredict; kwargs...)

    state, ll_increment = update(model, alg, iter, state, observation; kwargs...)
    isnothing(callback) ||
        callback(model, alg, iter, state, observation, PostUpdate; kwargs...)

    return state, (ll_increment - marginalization_term)
end

function initialise(
    rng::AbstractRNG,
    model::StateSpaceModel{T},
    filter::ParticleFilter;
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
    filter::ParticleFilter,
    step::Integer,
    state::ParticleDistribution,
    observation;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    proposed_particles = map(
        x -> simulate(
            rng, model, filter.proposal, step, x, observation; kwargs...
        ),
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
    filter::ParticleFilter,
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

# Default to latent dynamics
struct LatentProposal <: AbstractProposal end

const BootstrapFilter{RS} = ParticleFilter{RS,LatentProposal}
const BF = BootstrapFilter
BootstrapFilter(N::Integer; kwargs...) = ParticleFilter(N, LatentProposal(); kwargs...)

function simulate(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    prop::LatentProposal,
    step::Integer,
    state,
    observation;
    kwargs...,
)
    return SSMProblems.simulate(rng, model.dyn, step, state; kwargs...)
end

function logdensity(
    model::AbstractStateSpaceModel,
    prop::LatentProposal,
    step::Integer,
    prev_state,
    new_state,
    observation;
    kwargs...,
)
    return SSMProblems.logdensity(model.dyn, step, prev_state, new_state; kwargs...)
end

# overwrite predict for the bootstrap filter to remove redundant computation
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
    state.particles = map(
        x -> SSMProblems.simulate(rng, model.dyn, step, x; kwargs...), collect(state)
    )

    return update_ref!(state, ref_state, step)
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
