export BootstrapFilter, BF
export ParticleFilter, PF, AbstractProposal
export AuxiliaryParticleFilter, PredictivePosterior
export MeanPredictive, ModePredictive, DrawPredictive

import SSMProblems: distribution, simulate, logdensity

abstract type AbstractProposal end

function SSMProblems.distribution(
    prop::AbstractProposal, iter::Integer, state, observation; kwargs...
)
    return throw(MethodError(distribution, (prop, iter, state, observation, kwargs...)))
end

function SSMProblems.simulate(
    rng::AbstractRNG, prop::AbstractProposal, iter::Integer, state, observation; kwargs...
)
    return rand(rng, SSMProblems.distribution(prop, iter, state, observation; kwargs...))
end

function SSMProblems.logdensity(
    prop::AbstractProposal, iter::Integer, prev_state, new_state, observation; kwargs...
)
    return logpdf(
        SSMProblems.distribution(prop, iter, prev_state, observation; kwargs...), new_state
    )
end

abstract type AbstractParticleFilter <: AbstractFilter end
function num_particles end
function resampler end

function initialise(
    rng::AbstractRNG,
    prior::StatePrior,
    algo::AbstractParticleFilter;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    N = num_particles(algo)
    particles = map(1:N) do i
        ref = !isnothing(ref_state) && i == 1 ? ref_state[0] : nothing
        initialise_particle(rng, prior, algo, ref; kwargs...)
    end

    return ParticleDistribution(particles, false)
end

function predict(
    rng::AbstractRNG,
    dyn::LatentDynamics,
    algo::AbstractParticleFilter,
    iter::Integer,
    state,
    observation;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    particles = map(1:num_particles(algo)) do i
        particle = state.particles[i]
        ref = !isnothing(ref_state) && i == 1 ? ref_state[iter] : nothing
        predict_particle(rng, dyn, algo, iter, particle, observation, ref; kwargs...)
    end

    # Accumulate the baseline with LSE of weights after prediction (before update)
    # For plain PF/guided: ll_baseline is 0.0 on entry, becomes LSE_before
    # For APF with resample: ll_baseline already stores negative correction; add LSE_before
    return ParticleDistribution(
        particles, logsumexp(log_weights(state)) + state.ll_baseline
    )
end

function update(
    obs::ObservationProcess,
    algo::AbstractParticleFilter,
    iter::Integer,
    state::ParticleDistribution,
    observation;
    kwargs...,
)
    particles = map(state.particles) do particle
        update_particle(obs, algo, iter, particle, observation; kwargs...)
    end
    new_state, ll_increment = marginalise!(state, particles)

    return new_state, ll_increment
end

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

num_particles(algo::ParticleFilter) = algo.N
resampler(algo::ParticleFilter) = algo.resampler

function initialise_particle(
    rng::AbstractRNG, prior::StatePrior, algo::ParticleFilter, ref_state; kwargs...
)
    x = sample_prior(rng, prior, algo, ref_state; kwargs...)
    return Particle(x, 0)
end

function predict_particle(
    rng::AbstractRNG,
    dyn::LatentDynamics,
    algo::ParticleFilter,
    iter::Integer,
    particle::AbstractParticle,
    observation,
    ref_state;
    kwargs...,
)
    new_x, log_increment = propogate(
        rng, dyn, algo, iter, particle.state, observation, ref_state; kwargs...
    )
    return Particle(new_x, log_weight(particle) + log_increment, particle.ancestor)
end

function update_particle(
    obs::ObservationProcess,
    ::ParticleFilter,
    iter::Integer,
    particle::AbstractParticle,
    observation;
    kwargs...,
)
    log_increment = SSMProblems.logdensity(
        obs, iter, particle.state, observation; kwargs...
    )
    return Particle(particle.state, log_weight(particle) + log_increment, particle.ancestor)
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
    rs = resampler(algo)
    state = maybe_resample(rng, rs, state; ref_state)
    callback(model, algo, iter, state, observation, PostResample; kwargs...)
    return move(rng, model, algo, iter, state, observation; ref_state, callback, kwargs...)
end

function sample_prior(
    rng::AbstractRNG, prior::StatePrior, algo::ParticleFilter, ref_state; kwargs...
)
    x = if isnothing(ref_state)
        SSMProblems.simulate(rng, prior; kwargs...)
    else
        ref_state
    end
    return x
end

function propogate(
    rng::AbstractRNG,
    dyn,
    algo::ParticleFilter,
    iter::Integer,
    x,
    observation,
    ref_state;
    kwargs...,
)
    # TODO: use a trait to compute the sample and logpdf in one go if distribution is defined
    new_x = if isnothing(ref_state)
        SSMProblems.simulate(rng, algo.proposal, iter, x, observation; kwargs...)
    else
        ref_state
    end
    log_p = SSMProblems.logdensity(dyn, iter, x, new_x; kwargs...)
    log_q = SSMProblems.logdensity(algo.proposal, iter, x, new_x, observation; kwargs...)
    logw_inc = log_p - log_q
    return new_x, logw_inc
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

# overwrite propogate for the bootstrap filter to remove redundant computation
function propogate(
    rng::AbstractRNG,
    dyn,
    algo::BootstrapFilter,
    iter::Integer,
    x,
    observation,
    ref_state;
    kwargs...,
)
    new_x = if isnothing(ref_state)
        SSMProblems.simulate(rng, dyn, iter, x; kwargs...)
    else
        ref_state
    end

    # TODO: replace this with nothing (unweighted particle)
    return new_x, 0
end

# TODO: I feel like we shouldn't need to do this conversion. It should be handled by dispatch
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

abstract type PredictivePosterior end

struct AuxiliaryParticleFilter{PFT<:AbstractParticleFilter,PPT<:PredictivePosterior} <:
       AbstractFilter
    pf::PFT
    pp::PPT
end

resampler(algo::AuxiliaryParticleFilter) = resampler(algo.pf)
num_particles(algo::AuxiliaryParticleFilter) = num_particles(algo.pf)

function initialise(
    rng::AbstractRNG,
    prior::StatePrior,
    algo::AuxiliaryParticleFilter;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    return initialise(rng, prior, algo.pf; ref_state, kwargs...)
end

function step(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    algo::AuxiliaryParticleFilter,
    iter::Integer,
    state,
    observation;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    callback::CallbackType=nothing,
    kwargs...,
)
    # Compute lookahead weights approximating log p(y_{t+1} | x_{t}^(i))
    log_ξs = map(state.particles) do particle
        p_star = predictive_state(rng, dyn(model), algo, iter, particle; kwargs...)
        predictive_loglik(obs(model), algo.pf, iter, p_star, observation; kwargs...)
    end

    rs = AuxiliaryResampler(resampler(algo), log_ξs)
    state = maybe_resample(rng, rs, state; ref_state)

    callback(model, algo, iter, state, observation, PostResample; kwargs...)
    return move(
        rng, model, algo.pf, iter, state, observation; ref_state, callback, kwargs...
    )
end

struct MeanPredictive <: PredictivePosterior end

function predictive_statistic(
    ::AbstractRNG, ::MeanPredictive, dyn, iter::Integer, state; kwargs...
)
    transition_dist = SSMProblems.distribution(dyn, iter, state; kwargs...)
    return mean(transition_dist)
end

struct ModePredictive <: PredictivePosterior end

function predictive_statistic(
    ::AbstractRNG, ::ModePredictive, dyn, iter::Integer, state; kwargs...
)
    transition_dist = SSMProblems.distribution(dyn, iter, state; kwargs...)
    return mode(transition_dist)
end

struct DrawPredictive <: PredictivePosterior end

function predictive_statistic(
    rng::AbstractRNG, ::DrawPredictive, dyn, iter::Integer, state; kwargs...
)
    return SSMProblems.simulate(rng, dyn, iter, state; kwargs...)
end

# TODO (RB): Really these should be returning a state rather than a particle but we would
# need to define a RB state first
function predictive_state(
    rng::AbstractRNG,
    dyn::LatentDynamics,
    apf::AuxiliaryParticleFilter{<:AbstractParticleFilter},
    iter::Integer,
    particle::AbstractParticle;
    kwargs...,
)
    x_star = predictive_statistic(rng, apf.pp, dyn, iter, particle.state; kwargs...)
    return Particle(x_star, log_weight(particle), particle.ancestor)
end

function predictive_loglik(
    obs::ObservationProcess,
    algo::ParticleFilter,
    iter::Integer,
    p_star::Particle,
    observation;
    kwargs...,
)
    return SSMProblems.logdensity(obs, iter, p_star.state, observation; kwargs...)
end
