# # Kalman filter for a linear SSM 
using AbstractMCMC
using Distributions
using LinearAlgebra
using PDMats
using Plots
using Random
using SSMProblems

##########################
#### MODEL DEFINITION ####
##########################

struct LinearGaussianLatentDynamics{T<:Real} <: LatentDynamics
    """
        Latent dynamics for a linear Gaussian state space model.

        The model is defined by the following equations:
        x[0] = z + ϵ,                 ϵ    ∼ N(0, P)
        x[k] = Φx[k-1] + b + w[k],    w[k] ∼ N(0, Q)
    """
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

struct LinearGaussianObservationProcess{T<:Real} <: ObservationProcess
    """
        Observation process for a linear Gaussian state space model.

        The model is defined by the following equation:
        y[k] = Hx[k] + v[k],          v[k] ∼ N(0, R)
    """
    H::Matrix{T}
    R::PDMat{T}
end
function LinearGaussianObservationProcess(H::Matrix, R::Matrix)
    return LinearGaussianObservationProcess(H, PDMat(R))
end

# Define general transition and observation distributions to be used in forward simulation
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

# Define getter methods for Kalman filter parameters
calc_z(model::LinearGaussianLatentDynamics, step::Int, extra::Nothing) = model.z
calc_P(model::LinearGaussianLatentDynamics, step::Int, extra::Nothing) = model.P
calc_Φ(model::LinearGaussianLatentDynamics, step::Int, extra::Nothing) = model.Φ
calc_b(model::LinearGaussianLatentDynamics, step::Int, extra::Nothing) = model.b
calc_Q(model::LinearGaussianLatentDynamics, step::Int, extra::Nothing) = model.Q
calc_H(model::LinearGaussianObservationProcess, step::Int, extra::Nothing) = model.H
calc_R(model::LinearGaussianObservationProcess, step::Int, extra::Nothing) = model.R

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
    x_filts = Vector{typeof(z)}(undef, T)
    P_filts = Vector{typeof(P)}(undef, T)

    for t in 1:T
        # Extract parameters
        z = calc_z(model.latent_dynamics, t, extras[t])
        P = calc_P(model.latent_dynamics, t, extras[t])
        Φ = calc_Φ(model.latent_dynamics, t, extras[t])
        b = calc_b(model.latent_dynamics, t, extras[t])
        Q = calc_Q(model.latent_dynamics, t, extras[t])
        H = calc_H(model.observation_process, t, extras[t])
        R = calc_R(model.observation_process, t, extras[t])

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

function AbstractMCMC.sample(
    model::StateSpaceModel, observations::AbstractVector, algorithm
)
    return sample(model, observations, [nothing for _ in 1:length(observations)], algorithm)
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
p = plot(1:T, first.(xs); label="Truth")
scatter!(p, 1:T, first.(ys); label="Observations")
plot!(p, 1:T, first.(x_filts); ribbon=sqrt.(first.(P_filts)), label="Filtered")
