export BootstrapFilter, BF
export ParticleFilter, PF, AbstractProposal
export AuxiliaryParticleFilter
export AbstractLookAheadScore, RepresentativeStateLookAhead
export PredictiveStatistic, MeanPredictive, ModePredictive, DrawPredictive

"""
    AbstractProposal

Implement `distribution(proposal, t, state, observation)`, or both `simulate` and
`logdensity`. For an ordinary particle filter `state` is the previous latent sample.
For an RBPF it is the full `RBState`: proposals can inspect both the outer sample and
inner filtering belief, but must draw and score only the new outer sample. The inner
filter is advanced by RBPF after this draw.
"""
abstract type AbstractProposal end

function distribution(prop::AbstractProposal, iter::Integer, state, observation)
    return throw(MethodError(distribution, (prop, iter, state, observation)))
end

function simulate(
    rng::AbstractRNG, prop::AbstractProposal, iter::Integer, state, observation
)
    dist = distribution(prop, iter, state, observation)
    return simulate_from_dist(rng, dist)
end

function logdensity(
    prop::AbstractProposal, iter::Integer, prev_state, new_state, observation
)
    return logpdf(distribution(prop, iter, prev_state, observation), new_state)
end

function num_particles end
function resampler end

function initialise(
    rng::AbstractRNG,
    prior::StatePrior,
    algo::AbstractParticleFilter;
    ref_state::Union{Nothing,AbstractVector}=nothing,
)
    N = num_particles(algo)
    particles = map(1:N) do i
        ref = !isnothing(ref_state) && i == 1 ? _trajectory_state(ref_state, 0) : nothing
        initialise_particle(rng, prior, algo, ref)
    end

    return ParticleDistribution(particles, TypelessZero())
end

function predict(
    rng::AbstractRNG,
    dyn::LatentDynamics,
    algo::AbstractParticleFilter,
    iter::Integer,
    state,
    observation;
    ref_state::Union{Nothing,AbstractVector}=nothing,
)
    particles = map(1:num_particles(algo)) do i
        particle = state.particles[i]
        ref = if !isnothing(ref_state) && i == 1
            _trajectory_state(ref_state, iter)
        else
            nothing
        end
        predict_particle(rng, dyn, algo, iter, particle, observation, ref)
    end

    # Preserve the incoming weight normalizer; guided proposal corrections belong
    # to the new weights and must not be subtracted from the evidence increment.
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
    observation,
)
    particles = map(state.particles) do particle
        update_particle(obs, algo, iter, particle, observation)
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
    N > 0 || throw(ArgumentError("particle count must be positive"))
    conditional_resampler = ESSResampler(threshold, resampler)
    return ParticleFilter(N, conditional_resampler, proposal)
end

num_particles(algo::ParticleFilter) = algo.N
resampler(algo::ParticleFilter) = algo.resampler

function initialise_particle(
    rng::AbstractRNG, prior::StatePrior, algo::ParticleFilter, ref_state
)
    x = sample_prior(rng, prior, algo, ref_state)
    return Particle(x, 0)
end

function predict_particle(
    rng::AbstractRNG,
    dyn::LatentDynamics,
    algo::ParticleFilter,
    iter::Integer,
    particle::Particle,
    observation,
    ref_state,
)
    new_x, log_increment = propagate(
        rng, dyn, algo, iter, particle.state, observation, ref_state
    )
    return Particle(new_x, log_weight(particle) + log_increment, particle.ancestor)
end

function update_particle(
    obs::ObservationProcess,
    ::ParticleFilter,
    iter::Integer,
    particle::Particle,
    observation,
)
    log_increment = logdensity(obs, iter, particle.state, observation)
    return Particle(particle.state, log_weight(particle) + log_increment, particle.ancestor)
end

function step(
    rng::AbstractRNG,
    model::StateSpaceModel,
    algo::AbstractParticleFilter,
    iter::Integer,
    state,
    observation;
    ref_state::Union{Nothing,AbstractVector}=nothing,
)
    rs = resampler(algo)
    state = maybe_resample(rng, rs, state; ref_state)
    return move(rng, model, algo, iter, state, observation; ref_state)
end

function sample_prior(rng::AbstractRNG, prior::StatePrior, algo::ParticleFilter, ref_state)
    x = if isnothing(ref_state)
        simulate(rng, prior)
    else
        ref_state
    end
    return x
end

function propagate(
    rng::AbstractRNG, dyn, algo::ParticleFilter, iter::Integer, x, observation, ref_state
)
    # TODO: use a trait to compute the sample and logpdf in one go if distribution is defined
    new_x = if isnothing(ref_state)
        simulate(rng, algo.proposal, iter, x, observation)
    else
        ref_state
    end
    log_p = logdensity(dyn, iter, outer_component(x), new_x)
    log_q = logdensity(algo.proposal, iter, x, new_x, observation)
    logw_inc = log_p - log_q
    return new_x, logw_inc
end

outer_component(x) = x
outer_component(x::RBState) = x.x

struct LatentProposal <: AbstractProposal end

const BootstrapFilter{RS} = ParticleFilter{RS,LatentProposal}
const BF = BootstrapFilter
BootstrapFilter(N::Integer; kwargs...) = ParticleFilter(N, LatentProposal(); kwargs...)

function simulate(
    rng::AbstractRNG,
    model::StateSpaceModel,
    prop::LatentProposal,
    iter::Integer,
    state,
    observation,
)
    return simulate(rng, model.dyn, iter, state)
end

function logdensity(
    model::StateSpaceModel,
    prop::LatentProposal,
    iter::Integer,
    prev_state,
    new_state,
    observation,
)
    return logdensity(model.dyn, iter, prev_state, new_state)
end

# overwrite propagate for the bootstrap filter to remove redundant computation
function propagate(
    rng::AbstractRNG, dyn, algo::BootstrapFilter, iter::Integer, x, observation, ref_state
)
    new_x = if isnothing(ref_state)
        simulate(rng, dyn, iter, outer_component(x))
    else
        ref_state
    end

    return new_x, TypelessZero()
end

abstract type AbstractLookAheadScore end

function compute_logeta(
    rng::AbstractRNG,
    weight_strategy::AbstractLookAheadScore,
    model::StateSpaceModel,
    algo,
    iter::Integer,
    state,
    observation,
)
    return throw(
        MethodError(
            compute_logeta, (rng, weight_strategy, model, algo, iter, state, observation)
        ),
    )
end

abstract type PredictiveStatistic end

struct RepresentativeStateLookAhead{PPT<:PredictiveStatistic} <: AbstractLookAheadScore
    pp::PPT
end

struct AuxiliaryParticleFilter{PFT<:AbstractParticleFilter,WT<:AbstractLookAheadScore} <:
       AbstractParticleFilter
    pf::PFT
    weight_strategy::WT
end

function AuxiliaryParticleFilter(pf::AbstractParticleFilter, pp::PredictiveStatistic)
    return AuxiliaryParticleFilter(pf, RepresentativeStateLookAhead(pp))
end

function compute_logeta(
    rng::AbstractRNG,
    weight_strategy::RepresentativeStateLookAhead,
    model::StateSpaceModel,
    algo,
    iter::Integer,
    state,
    observation,
)
    state_star = predictive_state(rng, model.dyn, weight_strategy, algo, iter, state)
    return predictive_loglik(model.obs, algo, iter, state_star, observation)
end

resampler(algo::AuxiliaryParticleFilter) = resampler(algo.pf)
num_particles(algo::AuxiliaryParticleFilter) = num_particles(algo.pf)

function initialise(
    rng::AbstractRNG,
    prior::StatePrior,
    algo::AuxiliaryParticleFilter;
    ref_state::Union{Nothing,AbstractVector}=nothing,
)
    return initialise(rng, prior, algo.pf; ref_state)
end

function step(
    rng::AbstractRNG,
    model::StateSpaceModel,
    algo::AuxiliaryParticleFilter,
    iter::Integer,
    state,
    observation;
    ref_state::Union{Nothing,AbstractVector}=nothing,
)
    rs = _step_resampler(rng, model, algo, iter, state, observation)
    state = maybe_resample(rng, rs, state; ref_state)
    return move(rng, model, algo, iter, state, observation; ref_state)
end

# Auxiliary weights affect ancestor proposals, not the filtering target used by AS/BS.
_refreshment_filter(algo::AbstractParticleFilter) = algo
_refreshment_filter(algo::AuxiliaryParticleFilter) = _refreshment_filter(algo.pf)

function _step_resampler(rng, model, algo::AbstractParticleFilter, iter, state, observation)
    return resampler(algo)
end

function _step_resampler(
    rng, model, algo::AuxiliaryParticleFilter, iter, state, observation
)
    log_ηs = map(state.particles) do particle
        compute_logeta(
            rng, algo.weight_strategy, model, algo.pf, iter, particle.state, observation
        )
    end
    return AuxiliaryResampler(resampler(algo), log_ηs)
end

function move(
    rng::AbstractRNG,
    model::StateSpaceModel,
    algo::AuxiliaryParticleFilter,
    iter::Integer,
    state,
    observation;
    kwargs...,
)
    return move(rng, model, algo.pf, iter, state, observation; kwargs...)
end

struct MeanPredictive <: PredictiveStatistic end

function predictive_statistic(::AbstractRNG, ::MeanPredictive, dyn, iter::Integer, state)
    transition_dist = distribution(dyn, iter, state)
    return mean(transition_dist)
end

struct ModePredictive <: PredictiveStatistic end

function predictive_statistic(::AbstractRNG, ::ModePredictive, dyn, iter::Integer, state)
    transition_dist = distribution(dyn, iter, state)
    return mode(transition_dist)
end

struct DrawPredictive <: PredictiveStatistic end

function predictive_statistic(rng::AbstractRNG, ::DrawPredictive, dyn, iter::Integer, state)
    return simulate(rng, dyn, iter, state)
end

function predictive_state(
    rng::AbstractRNG,
    dyn::LatentDynamics,
    weight_strategy::RepresentativeStateLookAhead,
    algo,
    iter::Integer,
    state,
)
    return predictive_statistic(rng, weight_strategy.pp, dyn, iter, state)
end

function predictive_loglik(
    obs::ObservationProcess, algo::ParticleFilter, iter::Integer, state, observation
)
    return logdensity(obs, iter, state, observation)
end
