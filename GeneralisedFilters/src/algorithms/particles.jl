export BootstrapFilter, BF
export ParticleFilter, PF, AbstractProposal

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
    callback::CallbackType=nothing,
    kwargs...,
)
    state = resample(rng, algo.resampler, state; ref_state)
    callback(model, algo, iter, state, observation, PostResample; kwargs...)
    return move(rng, model, algo, iter, state, observation; ref_state, callback, kwargs...)
end

function initialise(
    rng::AbstractRNG,
    model::StateSpaceModel,
    algo::ParticleFilter;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    particles = map(1:(algo.N)) do i
        if !isnothing(ref_state) && i == 1
            ref_state[0]
        else
            SSMProblems.simulate(rng, model.prior; kwargs...)
        end
    end

    return Particles(particles)
end

function predict(
    rng::AbstractRNG,
    model::StateSpaceModel,
    algo::ParticleFilter,
    iter::Integer,
    state,
    observation;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    proposed_particles = map(enumerate(state)) do (i, particle)
        if !isnothing(ref_state) && i == 1
            ref_state[iter]
        else
            simulate(rng, model, algo.proposal, iter, particle, observation; kwargs...)
        end
    end

    log_increments = map(zip(proposed_particles, state)) do (new_state, prev_state)
        log_f = SSMProblems.logdensity(model.dyn, iter, prev_state, new_state; kwargs...)

        log_q = SSMProblems.logdensity(
            model, algo.proposal, iter, prev_state, new_state, observation; kwargs...
        )

        (log_f - log_q)
    end

    state.particles = proposed_particles
    state = update_weights(state, log_increments)
    return state
end

function update(
    model::StateSpaceModel,
    algo::ParticleFilter,
    iter::Integer,
    state,
    observation;
    kwargs...,
)
    log_increments = map(
        x -> SSMProblems.logdensity(model.obs, iter, x, observation; kwargs...),
        state.particles,
    )

    state = update_weights(state, log_increments)
    ll_increment = marginalise!(state)

    return state, ll_increment
end

struct LatentProposal <: AbstractProposal end

const BootstrapFilter{RS} = ParticleFilter{RS,LatentProposal}
const BF = BootstrapFilter
BootstrapFilter(N::Integer; kwargs...) = ParticleFilter(N, LatentProposal(); kwargs...)

function SSMProblems.simulate(
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

function SSMProblems.logdensity(
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
    algo::BootstrapFilter,
    iter::Integer,
    state,
    observation=nothing;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    state.particles = map(enumerate(state)) do (i, particle)
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
        HierarchicalPrior(model.outer_prior, model.inner_model.prior),
        HierarchicalDynamics(model.outer_dyn, model.inner_model.dyn),
        HierarchicalObservations(model.inner_model.obs),
    )
    return filter(rng, ssm, algo, observations; ref_state=ref_state, kwargs...)
end
