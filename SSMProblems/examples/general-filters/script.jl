using SSMProblems
using Distributions
using StatsFuns
using Random
using AbstractMCMC
using LinearAlgebra
using UnPack
using Plots

## PARTICLE DISTRIBUTIONS ##################################################################

abstract type ParticleDistribution end

mutable struct Particles{PT} <: ParticleDistribution
    particles::Vector{PT}
end

mutable struct WeightedParticles{PT,WT<:Real} <: ParticleDistribution
    particles::Vector{PT}
    log_weights::Vector{WT}
end

mutable struct GaussianDistribution{XT,ΣT} <: ParticleDistribution
    x::XT
    Σ::ΣT
end

function ParticleDistribution(particles::AbstractVector)
    return Particles(particles)
end

function ParticleDistribution(particles::AbstractVector, log_weights::Vector{<:Real})
    return WeightedParticles(particles, log_weights)
end

weights(state::Particles) = zeros(length(state.particles))
weights(state::WeightedParticles) = softmax(state.log_weights)

function update_weights(state::Particles, log_weights)
    return WeightedParticles(state.particles, log_weights)
end

function update_weights(state::WeightedParticles, log_weights)
    state.log_weights += log_weights
    return state
end

## BOOTSTRAP FILTER ########################################################################

abstract type AbstractFilter end

struct BootstrapFilter <: AbstractFilter
    N::Int
end

const BF = BootstrapFilter

function initialize(rng::AbstractRNG, model::StateSpaceModel, algo::BF; kwargs...)
    particles = map(1:(algo.N)) do _
        SSMProblems.simulate(rng, model.prior; kwargs...)
    end

    return ParticleDistribution(particles)
end

function resample(rng::AbstractRNG, algo::BF, state::ParticleDistribution)
    ws = weights(state)
    ess = inv(sum(abs2, ws))
    if ess <= algo.N * 0.5
        idx = rand(rng, Distributions.Categorical(ws), algo.N)

        # creates a new particle distribution in GF anyways
        return ParticleDistribution(state.particles[idx])
    else
        return state
    end
end

function predict(
    rng::AbstractRNG, model::StateSpaceModel, algo::BF, iter::Int, state; kwargs...
)
    state.particles = map(state.particles) do particle
        SSMProblems.simulate(rng, model.dyn, iter, particle; kwargs...)
    end
    return state
end

function update(model::StateSpaceModel, algo::BF, iter::Int, state, observation; kwargs...)
    log_increments = map(state.particles) do particle
        SSMProblems.logdensity(model.obs, iter, particle, observation; kwargs...)
    end
    return update_weights(state, log_increments)
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::StateSpaceModel,
    algo::AbstractFilter,
    observations::AbstractVector;
    kwargs...,
)
    T = length(observations)
    state = initialize(rng, model, algo; kwargs...)
    filtererd_states = []

    for t in 1:T
        state = resample(rng, algo, state)
        state = predict(rng, model, algo, t, state; kwargs...)
        state = update(model, algo, t, state, observations[t]; kwargs...)
        push!(filtererd_states, deepcopy(state))
    end

    return filtererd_states
end

## KALMAN FILTER ###########################################################################

struct KalmanFilter <: AbstractFilter end

const KF = KalmanFilter

function initialize(rng::AbstractRNG, model::StateSpaceModel, algo::KF; kwargs...)
    @unpack x, Σ = model.prior
    return GaussianDistribution(x, Σ)
end

function resample(::AbstractRNG, ::KF, state::GaussianDistribution)
    return state
end

function predict(
    ::AbstractRNG, model::StateSpaceModel, algo::KF, iter::Int, state; kwargs...
)
    @unpack Φ, b, Q = model.dyn
    x = Φ * state.x + b
    Σ = Φ * state.Σ * Φ' + Q
    return GaussianDistribution(x, Σ)
end

function update(model::StateSpaceModel, algo::KF, iter::Int, state, observation; kwargs...)
    @unpack H, R = model.obs
    K = state.Σ * H' / (H * state.Σ * H' + R)
    state.x = state.x + K * (observation - H * state.x)
    state.Σ = state.Σ - K * H * state.Σ
    return state
end

## STATE SPACE MODEL #######################################################################

struct GaussianPrior{XT<:AbstractVector,ΣT<:AbstractMatrix} <: StatePrior
    x::XT
    Σ::ΣT
end

struct LinearGaussianLatentDynamics{
    ΦT<:AbstractMatrix,bT<:AbstractArray,QT<:AbstractMatrix
} <: LatentDynamics
    Φ::ΦT
    b::bT
    Q::QT
end

struct LinearGaussianObservationProcess{HT<:AbstractMatrix,RT<:AbstractMatrix} <:
       ObservationProcess
    H::HT
    R::RT
end

function SSMProblems.distribution(prior::GaussianPrior; kwargs...)
    return MvNormal(prior.x, prior.Σ)
end

function SSMProblems.distribution(
    model::LinearGaussianLatentDynamics, step::Int, prev_state::AbstractVector; kwargs...
)
    return MvNormal(model.Φ * prev_state + model.b, model.Q)
end

function SSMProblems.distribution(
    model::LinearGaussianObservationProcess, step::Int, state::AbstractVector; kwargs...
)
    return MvNormal(model.H * state, model.R)
end

const LinearGaussianSSM = StateSpaceModel{
    <:GaussianPrior,<:LinearGaussianLatentDynamics,<:LinearGaussianObservationProcess
}

## DEMONSTRATION ###########################################################################

toy_model = StateSpaceModel(
    GaussianPrior([-1.0, 1.0], Matrix(1.0I, 2, 2)),
    LinearGaussianLatentDynamics([0.8 0.2; -0.1 0.8], zeros(2), [0.2 0.0; 0.0 0.5]),
    LinearGaussianObservationProcess([1.0 0.0;], Matrix(0.3I, 1, 1)),
);

rng = MersenneTwister(1234);
xs, ys = sample(rng, toy_model, 100);

kf_states = AbstractMCMC.sample(rng, toy_model, KalmanFilter(), ys);
bf_states = AbstractMCMC.sample(rng, toy_model, BootstrapFilter(1024), ys);

begin
    plt = plot(; title="First Dimension Filtered Estimates", xlabel="Step", ylabel="Value")
    scatter!(plt, first.(ys); label="Observations", ms=2)
    plot!(plt, first.(getproperty.(kf_states, :x)); label="Kalman Filtered")
    plot!(
        plt, first.(mean.(getproperty.(bf_states, :particles))); label="Bootstrap Filtered"
    )
    plt
end