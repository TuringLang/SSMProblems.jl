"""Implementation of a Kalman filter using the SSMProblems.jl interface."""

using AbstractMCMC
using Distributions
using LinearAlgebra
using PDMats
using Plots
using Random
using UnPack

using SSMProblems

##########################
#### MODEL DEFINITION ####
##########################

"""
Latent dynamics for a linear Gaussian state space model.

The model is defined by the following equations:
x[0] = z + ϵ,                 ϵ    ∼ N(0, P)
x[k] = Φx[k-1] + b + w[k],    w[k] ∼ N(0, Q)
"""
struct LinearGaussianLatentDynamics{T<:Real} <: LatentDynamics
    z::Vector{T}
    P::PDMat{T}
    Φ::Matrix{T}
    b::Vector{T}
    Q::PDMat{T}
end
# Convert covariance matrices to PDMats to avoid recomputing Cholesky factorizations
function LinearGaussianLatentDynamics(z::Vector, P::Matrix, Φ::Matrix, b::Vector, Q::Matrix)
    return LinearGaussianLatentDynamics(z, PDMat(P), Φ, b, PDMat(Q))
end

"""
    Observation process for a linear Gaussian state space model.

    The model is defined by the following equation:
    y[k] = Hx[k] + v[k],          v[k] ∼ N(0, R)
"""
struct LinearGaussianObservationProcess{T<:Real} <: ObservationProcess
    H::Matrix{T}
    R::PDMat{T}
end
function LinearGaussianObservationProcess(H::Matrix, R::Matrix)
    return LinearGaussianObservationProcess(H, PDMat(R))
end

# Define general transition and observation distributions to be used in forward simulation.
# Since our model is homogenous (doesn't depend on control variables), we use `Nothing` for
# the type of `extra`.
function SSMProblems.initialisation_distribution(
    model::LinearGaussianLatentDynamics, extra::Nothing
)
    return MvNormal(model.z, model.P)
end
function SSMProblems.transition_distribution(
    model::LinearGaussianLatentDynamics{T},
    state::AbstractVector{T},
    step::Int,
    extra::Nothing,
) where {T}
    return MvNormal(model.Φ * state + model.b, model.Q)
end
function SSMProblems.observation_distribution(
    model::LinearGaussianObservationProcess{T},
    state::AbstractVector{T},
    step::Int,
    extra::Nothing,
) where {T}
    return MvNormal(model.H * state, model.R)
end

#######################
#### KALMAN FILTER ####
#######################

struct KalmanFilter end

const LinearGaussianSSM = StateSpaceModel{
    <:LinearGaussianLatentDynamics,<:LinearGaussianObservationProcess
}

function AbstractMCMC.sample(
    model::LinearGaussianSSM,
    observations::AbstractVector,
    extras::AbstractVector,
    ::KalmanFilter,
)
    T = length(observations)
    latent_type = only(typeof(model.latent_dynamics).parameters)
    x_filts = Vector{Vector{latent_type}}(undef, T)
    P_filts = Vector{Matrix{latent_type}}(undef, T)

    # Extract parameters
    @unpack z, P, Φ, b, Q = model.latent_dynamics
    @unpack H, R = model.observation_process

    for t in 1:T
        # Prediction step
        x_pred, P_pred = if t == 1
            z, P
        else
            Φ * x_filts[t - 1] + b, Φ * P_filts[t - 1] * Φ' + Q
        end

        # Update step
        y = observations[t]
        K = P_pred * H' / (H * P_pred * H' + R)
        x_filt = x_pred + K * (y - H * x_pred)
        P_filt = P_pred - K * H * P_pred

        x_filts[t] = x_filt
        P_filts[t] = P_filt
    end

    return x_filts, P_filts
end

# If no extras are provided, assume they are all `nothing`
function AbstractMCMC.sample(
    model::StateSpaceModel, observations::AbstractVector, algorithm
)
    extras = fill(nothing, length(observations))
    return sample(model, observations, extras, algorithm)
end

##################################
#### SIMULATION AND FILTERING ####
##################################

# Simulation parameters
SEED = 1
T = 100
z = [-1.0, 1.0]
P = Matrix(1.0I, 2, 2)
Φ = [0.8 0.2; -0.1 0.8]
b = zeros(2)
Q = [0.2 0.0; 0.0 0.5]
H = [1.0 0.0;]
R = Matrix(0.3I, 1, 1)

dyn = LinearGaussianLatentDynamics(z, P, Φ, b, Q)
obs = LinearGaussianObservationProcess(H, R)
model = StateSpaceModel(dyn, obs)

# Generate synthetic data
rng = MersenneTwister(SEED)
xs, ys = sample(rng, model, T)

# Run Kalman filter
x_filts, P_filts = AbstractMCMC.sample(model, ys, KalmanFilter())

# Plot trajectory for first dimension
p = plot(; title="First Dimension Kalman Filter Estimates", xlabel="Step", ylabel="Value")
plot!(p, 1:T, first.(xs); label="Truth")
scatter!(p, 1:T, first.(ys); label="Observations")
plot!(
    p,
    1:T,
    first.(x_filts);
    ribbon=sqrt.(first.(P_filts)),
    label="Filtered (±1σ)",
    fillalpha=0.2,
)
