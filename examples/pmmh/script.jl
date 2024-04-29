"""
Particle-Marginal Metropolis-Hasting using Turing.jl and AdvancedPS.jl.

This script demonstrates how the SSMProblems interface can be used with AdvancedPS.jl and
Turing.jl to perform particle-marginal Metropolis-Hastings as introduced in Andrieu et al.
(2010).

We test this algorithm on a simple linear Gaussian state-space model with an unknown drift
parameter. This drift parameter can be included in an augmented state vector and filtered
exactly using a Kalman filter, allowing us to compare the PMMH results to the ground truth
posterior.

The dynamics of the model are defined as follows:

    x_t = x_{t-1} + b + w_t, w_t ~ N(0, σ_x^2)
    y_t = x_t + v_t, v_t ~ N(0, σ_y^2)

The augmented model has state vector z_t = [x_t, b_t].
"""

using AdvancedPS
using AdvancedMH
using Distributions
using DynamicIterators
using GaussianDistributions
using Kalman
using LinearAlgebra
using PDMats
using Random
using Statistics
using StableRNGs
using SSMProblems
using Turing

# Model parameters
x_prior_mean = 0.0
x_prior_std = 1.0
b_prior_mean = 0.5
b_prior_std = 1.0
σ_x = 0.1
σ_y = 0.1
T = 5

# Sampling parameters
N_particles = 200
N_samples = 1000
N_burnin = 1000

# Define the augmented model using the conventions of Kalman.jl
z0 = [x_prior_mean, b_prior_mean]
P0 = Diagonal([x_prior_std^2, b_prior_std^2])
Φ = [1.0 1.0; 0.0 1.0]
b = zeros(2)  # in augmented model there is no drift
Q = PDiagMat([σ_x^2, 0.0])  # no noise on drift term

H = [1.0 0.0]
R = Diagonal([σ_y^2])

# Kalman.jl model
E = LinearEvolution(Φ, Gaussian(b, Q))
Obs = LinearObservationModel(H, R)
G0 = Gaussian(z0, P0)
M = LinearStateSpaceModel(E, Obs)
O = LinearObservation(E, H, R)

# Simulate from model
rng = StableRNG(1234)
initial = rand(rng, StateObs(G0, M.obs))
trajectory = trace(DynamicIterators.Sampled(M, rng), 1 => initial, endtime(T))
y_pairs = collect(t => y for (t, (x, y)) in pairs(trajectory))
ys = [y[1] for (t, (x, y)) in pairs(trajectory)]

# SSMProblems.jl model, defined conditionally on a specific value of the drift parameter
struct LinearGaussianParams
    x_prior_mean::Float64
    x_prior_std::Float64
    drift::Float64
    σ_x::Float64
    σ_y::Float64
end

mutable struct LinearGaussianModel <: SSMProblems.AbstractStateSpaceModel
    X::Vector{Float64}
    observations::Vector{Float64}
    θ::LinearGaussianParams
    function LinearGaussianModel(y::Vector{Float64}, θ::LinearGaussianParams)
        return new(Vector{Float64}(), y, θ)
    end
end

function SSMProblems.transition!!(rng::AbstractRNG, model::LinearGaussianModel)
    return rand(rng, Normal(model.θ.x_prior_mean, model.θ.x_prior_std))
end
function SSMProblems.transition!!(
    rng::AbstractRNG, model::LinearGaussianModel, state::Float64, step::Int
)
    return rand(rng, Normal(state + model.θ.drift, model.θ.σ_x))
end
function SSMProblems.transition_logdensity(
    model::LinearGaussianModel, prev_state::Float64, curr_state::Float64, step::Integer
)
    return logpdf(Normal(prev_state + model.θ.drift, model.θ.σ_x), curr_state)
end
function SSMProblems.emission_logdensity(
    model::LinearGaussianModel, state::Float64, step::Integer
)
    return logpdf(Normal(state, model.θ.σ_y), model.observations[step])
end
AdvancedPS.isdone(::LinearGaussianModel, step::Int) = step > T

# Define the Turing.jl model
pgas = AdvancedPS.PGAS(N_particles)
@model function pmmh_model(y)
    b ~ Normal(b_prior_mean, b_prior_std)

    params = LinearGaussianParams(x_prior_mean, x_prior_std, b, σ_x, σ_y)
    model = LinearGaussianModel(y, params)

    chain = only(sample(rng, model, pgas, 1))
    log_evidence = chain.logevidence

    Turing.@addlogprob! log_evidence

    return chain.trajectory.model.X
end

# Sample from the model
chn_marg = sample(pmmh_model(ys), MH(), N_samples; discard_initial=N_burnin, progress=true)

# Compare to ground truth
Xf, ll = kalmanfilter(O, 0 => Gaussian(z0, P0), y_pairs)
println("True filtered drift mean: ", Xf.x[end].μ[2])
println("PMMH filtered drift mean: ", mean(chn_marg[:b].data))
