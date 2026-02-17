export BootstrapFilter, BF
export ParticleFilter, PF, AbstractProposal
export AuxiliaryParticleFilter
export AbstractLookAheadScore, RepresentativeStateLookAhead
export PredictiveStatistic, MeanPredictive, ModePredictive, DrawPredictive
export AbstractThreadingStrategy, SingleThreaded, MultiThreaded

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
    dist = SSMProblems.distribution(prop, iter, state, observation; kwargs...)
    return SSMProblems.simulate_from_dist(rng, dist)
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
function threading_strategy end
function make_thread_rng end

abstract type AbstractThreadingStrategy end
struct SingleThreaded <: AbstractThreadingStrategy end
struct MultiThreaded <: AbstractThreadingStrategy
    min_batch_size::Int
    nworkers::Int
end
function MultiThreaded(; min_batch_size::Integer=2_048, nworkers::Union{Nothing,Integer}=nothing)
    if min_batch_size < 1
        throw(ArgumentError("min_batch_size must be at least 1"))
    end
    if !isnothing(nworkers) && (nworkers < 1)
        throw(ArgumentError("nworkers must be at least 1 when provided"))
    end
    nworkers_int = isnothing(nworkers) ? 0 : Int(nworkers)
    return MultiThreaded(Int(min_batch_size), nworkers_int)
end

threading_strategy(::AbstractParticleFilter) = SingleThreaded()
make_thread_rng(::AbstractRNG, seed::UInt64) = Random.Xoshiro(seed)
make_thread_rng(::StableRNGs.StableRNG, seed::UInt64) = StableRNGs.StableRNG(seed)

function _effective_nworkers(strategy::MultiThreaded)
    available = Base.Threads.nthreads()
    if strategy.nworkers == 0
        return available
    end
    return min(strategy.nworkers, available)
end

function _chunk_bounds(N::Integer, worker::Integer, nworkers::Integer)
    lo = ((worker - 1) * N) ÷ nworkers + 1
    hi = (worker * N) ÷ nworkers
    return lo, hi
end

function _map_particle_indices(f, ::SingleThreaded, N::Integer)
    return map(f, 1:N)
end

function _map_particle_indices(f, strategy::MultiThreaded, N::Integer)
    if N <= 0
        return []
    end

    nworkers = _effective_nworkers(strategy)
    if (nworkers == 1) || (N < strategy.min_batch_size)
        return map(f, 1:N)
    end

    x1 = f(1)
    xs = Vector{typeof(x1)}(undef, N)
    xs[1] = x1

    nworkers = min(nworkers, N)
    @sync for worker in 1:nworkers
        Base.Threads.@spawn begin
            lo, hi = _chunk_bounds(N, worker, nworkers)
            start = max(lo, 2)
            @inbounds for i in start:hi
                xs[i] = f(i)
            end
        end
    end
    return xs
end

function _map_particle_indices(f, rng::AbstractRNG, ::SingleThreaded, N::Integer)
    return map(i -> f(rng, i), 1:N)
end

function _map_particle_indices(
    f, rng::AbstractRNG, strategy::MultiThreaded, N::Integer
)
    if N <= 0
        return []
    end

    nworkers = _effective_nworkers(strategy)
    if (nworkers == 1) || (N < strategy.min_batch_size)
        return map(i -> f(rng, i), 1:N)
    end

    nworkers = min(nworkers, N)
    thread_seeds = rand(rng, UInt64, nworkers)
    thread_rngs = map(seed -> make_thread_rng(rng, seed), thread_seeds)

    x1 = f(thread_rngs[1], 1)
    xs = Vector{typeof(x1)}(undef, N)
    xs[1] = x1

    @sync for worker in 1:nworkers
        Base.Threads.@spawn begin
            lo, hi = _chunk_bounds(N, worker, nworkers)
            start = max(lo, 2)
            local_rng = thread_rngs[worker]
            @inbounds for i in start:hi
                xs[i] = f(local_rng, i)
            end
        end
    end
    return xs
end

function initialise(
    rng::AbstractRNG,
    prior::StatePrior,
    algo::AbstractParticleFilter;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    N = num_particles(algo)
    strategy = threading_strategy(algo)
    particles = _map_particle_indices(rng, strategy, N) do local_rng, i
        ref = !isnothing(ref_state) && i == 1 ? ref_state[0] : nothing
        initialise_particle(local_rng, prior, algo, ref; kwargs...)
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
    kwargs...,
)
    strategy = threading_strategy(algo)
    particles = _map_particle_indices(rng, strategy, num_particles(algo)) do local_rng, i
        particle = state.particles[i]
        ref = !isnothing(ref_state) && i == 1 ? ref_state[iter] : nothing
        predict_particle(local_rng, dyn, algo, iter, particle, observation, ref; kwargs...)
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
    strategy = threading_strategy(algo)
    particles = _map_particle_indices(strategy, length(state.particles)) do i
        particle = state.particles[i]
        update_particle(obs, algo, iter, particle, observation; kwargs...)
    end
    new_state, ll_increment = marginalise!(state, particles)

    return new_state, ll_increment
end

struct ParticleFilter{RS,PT,TS<:AbstractThreadingStrategy} <: AbstractParticleFilter
    N::Int
    resampler::RS
    proposal::PT
    threading::TS
end

const PF = ParticleFilter

function ParticleFilter(
    N::Integer,
    proposal::PT;
    threshold::Real=1.0,
    resampler::AbstractResampler=Systematic(),
    threading::AbstractThreadingStrategy=SingleThreaded(),
) where {PT<:AbstractProposal}
    conditional_resampler = ESSResampler(threshold, resampler)
    return ParticleFilter(N, conditional_resampler, proposal, threading)
end

ParticleFilter(N::Integer, rs, proposal) = ParticleFilter(N, rs, proposal, SingleThreaded())

num_particles(algo::ParticleFilter) = algo.N
resampler(algo::ParticleFilter) = algo.resampler
threading_strategy(algo::ParticleFilter) = algo.threading

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
    particle::Particle,
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
    particle::Particle,
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
    algo::ParticleFilter{RS,PT,TS},
    iter::Integer,
    x,
    observation,
    ref_state;
    kwargs...,
) where {RS,PT<:AbstractProposal,TS}
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

const BootstrapFilter{RS,TS} = ParticleFilter{RS,LatentProposal,TS}
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
    algo::ParticleFilter{RS,LatentProposal,TS},
    iter::Integer,
    x,
    observation,
    ref_state;
    kwargs...,
) where {RS,TS}
    new_x = if isnothing(ref_state)
        SSMProblems.simulate(rng, dyn, iter, x; kwargs...)
    else
        ref_state
    end

    return new_x, TypelessZero()
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

abstract type AbstractLookAheadScore end

function compute_logeta(
    rng::AbstractRNG,
    weight_strategy::AbstractLookAheadScore,
    model::AbstractStateSpaceModel,
    algo,
    iter::Integer,
    state,
    observation;
    kwargs...,
)
    return throw(
        MethodError(
            compute_logeta,
            (rng, weight_strategy, model, algo, iter, state, observation, kwargs...),
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
    model::AbstractStateSpaceModel,
    algo,
    iter::Integer,
    state,
    observation;
    kwargs...,
)
    state_star = predictive_state(
        rng, dyn(model), weight_strategy, algo, iter, state; kwargs...
    )
    return predictive_loglik(obs(model), algo, iter, state_star, observation; kwargs...)
end

resampler(algo::AuxiliaryParticleFilter) = resampler(algo.pf)
num_particles(algo::AuxiliaryParticleFilter) = num_particles(algo.pf)
threading_strategy(algo::AuxiliaryParticleFilter) = threading_strategy(algo.pf)

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
    strategy = threading_strategy(algo)
    log_ηs = _map_particle_indices(rng, strategy, length(state.particles)) do local_rng, i
        particle = state.particles[i]
        compute_logeta(
            local_rng,
            algo.weight_strategy,
            model,
            algo.pf,
            iter,
            particle.state,
            observation;
            kwargs...,
        )
    end

    rs = AuxiliaryResampler(resampler(algo), log_ηs)
    state = maybe_resample(rng, rs, state; ref_state)

    callback(model, algo, iter, state, observation, PostResample; kwargs...)
    return move(
        rng, model, algo.pf, iter, state, observation; ref_state, callback, kwargs...
    )
end

struct MeanPredictive <: PredictiveStatistic end

function predictive_statistic(
    ::AbstractRNG, ::MeanPredictive, dyn, iter::Integer, state; kwargs...
)
    transition_dist = SSMProblems.distribution(dyn, iter, state; kwargs...)
    return mean(transition_dist)
end

struct ModePredictive <: PredictiveStatistic end

function predictive_statistic(
    ::AbstractRNG, ::ModePredictive, dyn, iter::Integer, state; kwargs...
)
    transition_dist = SSMProblems.distribution(dyn, iter, state; kwargs...)
    return mode(transition_dist)
end

struct DrawPredictive <: PredictiveStatistic end

function predictive_statistic(
    rng::AbstractRNG, ::DrawPredictive, dyn, iter::Integer, state; kwargs...
)
    return SSMProblems.simulate(rng, dyn, iter, state; kwargs...)
end

function predictive_state(
    rng::AbstractRNG,
    dyn::LatentDynamics,
    weight_strategy::RepresentativeStateLookAhead,
    algo,
    iter::Integer,
    state;
    kwargs...,
)
    return predictive_statistic(rng, weight_strategy.pp, dyn, iter, state; kwargs...)
end

function predictive_loglik(
    obs::ObservationProcess,
    algo::ParticleFilter,
    iter::Integer,
    state,
    observation;
    kwargs...,
)
    return SSMProblems.logdensity(obs, iter, state, observation; kwargs...)
end
