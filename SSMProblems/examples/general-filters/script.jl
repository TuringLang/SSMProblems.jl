using SSMProblems
using Distributions
using StatsFuns
using Random
using AbstractMCMC
using LinearAlgebra
using UnPack
using Plots
using StatsBase

## UTILITIES ###############################################################################

function logmeanexp(x::AbstractArray{T}; dims=:)::T where {T}
    max_ = @fastmath reduce(max, x; dims, init=float(T)(-Inf))
    @fastmath max_ .+ log.(mean(exp.(x .- max_); dims))
end

# for general PF, KF, and BF in that order
log_marginal(log_trans_inc, log_obs_inc) = logmeanexp(log_trans_inc + log_obs_inc)
log_marginal(::Nothing, log_obs_inc::Real) = log_obs_inc
log_marginal(::Nothing, log_obs_inc::AbstractVector) = logmeanexp(log_obs_inc)

## PARTICLE DISTRIBUTIONS ##################################################################

mutable struct UniformParticles{PT}
    particles::Vector{PT}
end

mutable struct ParticleDistribution{PT,WT<:Real}
    particles::Vector{PT}
    log_weights::Vector{WT}
end

struct GaussianDistribution{PT,ΣT}
    μ::PT
    Σ::ΣT
end

StatsBase.weights(state::ParticleDistribution) = Weights(softmax(state.log_weights))

function update_weights!(state::UniformParticles, log_weights)
    return ParticleDistribution(state.particles, log_weights)
end

function update_weights!(state::ParticleDistribution, log_weights)
    state.log_weights += log_weights
    return state
end

StatsBase.mean(states::GaussianDistribution) = states.μ
StatsBase.mean(states::ParticleDistribution) = mean(states.particles, weights(states))

## FILTERING INTERFACE #####################################################################

abstract type AbstractFilter end

resample(::AbstractRNG, ::AbstractFilter, state) = state

function step(
    rng::AbstractRNG,
    model::StateSpaceModel,
    algo::AbstractFilter,
    iter::Int,
    state,
    observation;
    kwargs...,
)
    state = resample(rng, algo, state)
    state, log_inc_1 = predict(rng, model, algo, iter, state, observation; kwargs...)
    state, log_inc_2 = update(model, algo, iter, state, observation; kwargs...)
    return state, log_marginal(log_inc_1, log_inc_2)
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::StateSpaceModel,
    algo::AbstractFilter,
    observations::AbstractVector;
    kwargs...,
)
    init_state = initialize(rng, model, algo; kwargs...)
    state, log_evidence = step(rng, model, algo, 1, init_state, observations[1]; kwargs...)

    T = length(observations)
    filtererd_states = fill(deepcopy(state), T)

    for t in 2:T
        state, log_marginal = step(rng, model, algo, t, state, observations[t]; kwargs...)
        log_evidence += log_marginal
        filtererd_states[t] = deepcopy(state)
    end
    return filtererd_states, log_evidence
end

## PROPOSALS ###############################################################################

abstract type AbstractProposal end

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

## PARTICLE FILTER #########################################################################

struct ParticleFilter{PT} <: AbstractFilter
    N::Int
    threshold::Float64
    proposal::PT
end
const PF = ParticleFilter
ParticleFilter(N::Int, proposal; threshold=1.0) = ParticleFilter(N, threshold, proposal)

function initialize(rng::AbstractRNG, model::StateSpaceModel, algo::PF; kwargs...)
    particles = map(1:(algo.N)) do _
        SSMProblems.simulate(rng, model.prior; kwargs...)
    end
    return UniformParticles(particles)
end

function resample(rng::AbstractRNG, algo::PF, state::ParticleDistribution)
    ws = weights(state)
    ess = inv(sum(abs2, ws))
    if ess <= algo.N * algo.threshold
        idx = rand(rng, Distributions.Categorical(ws), algo.N)
        return ParticleDistribution(state.particles[idx], zero(ws))
    else
        return state
    end
end

function predict(
    rng::AbstractRNG,
    model::StateSpaceModel,
    algo::PF,
    iter::Int,
    state,
    observation;
    kwargs...,
)
    proposed_particles = map(state.particles) do particle
        SSMProblems.simulate(rng, model, algo.proposal, iter, particle, observation; kwargs...)
    end

    log_increments =
        map(zip(proposed_particles, state.particles)) do (new_state, prev_state)
            log_f = SSMProblems.logdensity(
                model.dyn, iter, prev_state, new_state; kwargs...
            )
            log_q = SSMProblems.logdensity(
                model, algo.proposal, iter, prev_state, new_state, observation; kwargs...
            )
            (log_f - log_q)
        end

    state.particles = proposed_particles
    return update_weights!(state, log_increments), log_increments
end

function update(model::StateSpaceModel, algo::PF, iter::Int, state, observation; kwargs...)
    log_increments = map(state.particles) do particle
        SSMProblems.logdensity(model.obs, iter, particle, observation; kwargs...)
    end
    return update_weights!(state, log_increments), log_increments
end

## BOOTSTRAP FILTER ########################################################################

const BootstrapFilter = ParticleFilter{Nothing}
const BF = BootstrapFilter
BootstrapFilter(N::Integer; kwargs...) = ParticleFilter(N, nothing; kwargs...)

function predict(
    rng::AbstractRNG,
    model::StateSpaceModel,
    algo::BF,
    iter::Int,
    state,
    observation;
    kwargs...,
)
    state.particles = map(state.particles) do particle
        SSMProblems.simulate(rng, model.dyn, iter, particle; kwargs...)
    end
    return state, nothing
end

## KALMAN FILTER ###########################################################################

struct KalmanFilter <: AbstractFilter end

const KF = KalmanFilter

function initialize(rng::AbstractRNG, model::StateSpaceModel, algo::KF; kwargs...)
    @unpack μ, Σ = model.prior
    return GaussianDistribution(μ, Σ)
end

function predict(
    ::AbstractRNG,
    model::StateSpaceModel,
    algo::KF,
    iter::Int,
    state,
    observation;
    kwargs...,
)
    @unpack Φ, b, Q = model.dyn
    state = GaussianDistribution(Φ * state.μ + b, Φ * state.Σ * Φ' + Q)
    return state, nothing
end

function update(model::StateSpaceModel, algo::KF, iter::Int, state, observation; kwargs...)
    @unpack H, c, R = model.obs
    m = H * state.μ + c
    S = hermitianpart(H * state.Σ * H' + R)
    K = state.Σ * H' / S

    state = GaussianDistribution(state.μ + K * (observation - m), state.Σ - K * H * state.Σ)
    return state, logpdf(MvNormal(m, S), observation)
end

## STATE SPACE MODEL #######################################################################

struct GaussianPrior{XT<:AbstractVector,ΣT<:AbstractMatrix} <: StatePrior
    μ::XT
    Σ::ΣT
end

struct LinearGaussianLatentDynamics{
    ΦT<:AbstractMatrix,bT<:AbstractVector,QT<:AbstractMatrix
} <: LatentDynamics
    Φ::ΦT
    b::bT
    Q::QT
end

struct LinearGaussianObservationProcess{
    HT<:AbstractMatrix,cT<:AbstractVector,RT<:AbstractMatrix
} <: ObservationProcess
    H::HT
    c::cT
    R::RT
end

SSMProblems.distribution(prior::GaussianPrior; kwargs...) = MvNormal(prior.μ, prior.Σ)

function SSMProblems.distribution(
    model::LinearGaussianLatentDynamics, step::Int, prev_state::AbstractVector; kwargs...
)
    return MvNormal(model.Φ * prev_state + model.b, model.Q)
end

function SSMProblems.distribution(
    model::LinearGaussianObservationProcess, step::Int, state::AbstractVector; kwargs...
)
    return MvNormal(model.H * state + model.c, model.R)
end

const LinearGaussianSSM = StateSpaceModel{
    <:GaussianPrior,<:LinearGaussianLatentDynamics,<:LinearGaussianObservationProcess
}

struct LinearGaussianProposal <: AbstractProposal end

function SSMProblems.distribution(
    model::AbstractStateSpaceModel,
    kernel::LinearGaussianProposal,
    iter::Int,
    state,
    observation;
    kwargs...,
)
    @unpack Φ, b, Q = model.dyn
    pred = GaussianDistribution(Φ * state + b, Q)
    prop, _ = update(model, KF(), iter, pred, observation; kwargs...)
    return MvNormal(prop.μ, hermitianpart(prop.Σ))
end

## DEMONSTRATION ###########################################################################

toy_model = StateSpaceModel(
    GaussianPrior([-1, 1], I(2)),
    LinearGaussianLatentDynamics([0.8 0.2; -0.1 0.8], zeros(2), Diagonal([0.2, 0.5])),
    LinearGaussianObservationProcess([1 0;], zeros(1), 0.3I(1)),
);

rng = MersenneTwister(1234);
xs, ys = sample(rng, toy_model, 100);
prop = LinearGaussianProposal();

kf_states, kf_ll = AbstractMCMC.sample(rng, toy_model, KF(), ys);
bf_states, bf_ll = AbstractMCMC.sample(rng, toy_model, BF(512; threshold=0.5), ys);
pf_states, pf_ll = AbstractMCMC.sample(rng, toy_model, PF(64, prop; threshold=0.5), ys);

begin
    plt = plot(; title="First Dimension Filtered Estimates")
    scatter!(plt, first.(ys); label=nothing, ms=2)
    plot!(plt, first.(mean.(kf_states)); label="Kalman Filter")
    plot!(plt, first.(mean.(bf_states)); label="Bootstrap Filter")
    plot!(plt, first.(mean.(pf_states)); label="Optimal Filter")
    plt
end
