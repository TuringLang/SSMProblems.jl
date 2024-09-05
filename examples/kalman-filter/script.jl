# # Kalman Filter
#
# This example implements a Kalman filter for a linear Gaussian state space model using the
# SSMProblems interface. 

using AbstractMCMC
using Distributions
using LinearAlgebra
using Plots
using Random
using UnPack

using SSMProblems

# ## Model Definition

# We start by defining structs to store the paramaters for our specific latent dynamics and
# observation process.
#
# The latent dynamics have the following form:
# ```
# x[0] = z + ϵ,                 ϵ    ∼ N(0, P)
# x[k] = Φx[k-1] + b + w[k],    w[k] ∼ N(0, Q)
# ```
#
# We store all of these paramaters in a struct.

struct LinearGaussianLatentDynamics{T<:Real} <: LatentDynamics{Vector{T}}
    z::Vector{T}
    P::Matrix{T}
    Φ::Matrix{T}
    b::Vector{T}
    Q::Matrix{T}
end

# Similarly, the observation process is defined by the following equation:
# ```
# y[k] = Hx[k] + v[k],          v[k] ∼ N(0, R)
# ```

struct LinearGaussianObservationProcess{T<:Real} <: ObservationProcess{Vector{T}}
    H::Matrix{T}
    R::Matrix{T}
end

# We then define general transition and observation distributions to be used in forward
# simulation. Since our model is homogenous (doesn't depend on control variables), we use
# `Nothing` for the type of `extra`.
#
# Even if we did not require forward simulation (e.g. we were given observations), it is
# still useful to define these methods as they allow us to run a particle filter on our
# model with no additional implementation required. Although a Kalman filter would generally
# be preferred in this linear Gaussian case, it may be of interest to compare the sampling
# performance with a general particle filter.

function SSMProblems.distribution(model::LinearGaussianLatentDynamics, extra::Nothing)
    return MvNormal(model.z, model.P)
end
function SSMProblems.distribution(
    model::LinearGaussianLatentDynamics{T}, step::Int, prev_state::Vector{T}, extra::Nothing
) where {T}
    return MvNormal(model.Φ * prev_state + model.b, model.Q)
end
function SSMProblems.distribution(
    model::LinearGaussianObservationProcess{T}, step::Int, state::Vector{T}, extra::Nothing
) where {T}
    return MvNormal(model.H * state, model.R)
end

# ## Filtering Algorithm

# We define a concrete type to represent our sampling algorithm. This is used for dispatch
# to, say, distinguish from using a generic particle filter.

struct KalmanFilter end

# A Kalman filter is only valid for linear Gaussian state space models, so we define an
# alias for an SSM with linear Gaussian latent dynamics and observation process, which will
# be used to dispatch to the correct method.

const LinearGaussianSSM{T} = StateSpaceModel{
    <:LinearGaussianLatentDynamics{T},<:LinearGaussianObservationProcess{T}
};

# We then define a method for the `sample` function. This is a standardised interface which
# requires the model we are sampling from, the sampling algorithm as well as the
# observations and any extras.

function AbstractMCMC.sample(
    model::LinearGaussianSSM{U},
    ::KalmanFilter,
    observations::AbstractVector,
    extra0,
    extras::AbstractVector,
) where {U}
    T = length(observations)
    x_filts = Vector{Vector{U}}(undef, T)
    P_filts = Vector{Matrix{U}}(undef, T)

    @unpack z, P, Φ, b, Q = model.dyn  ## Extract parameters
    @unpack H, R = model.obs

    ## Initialise the filter
    x = z
    P = P

    for t in 1:T
        ## Prediction step
        x = Φ * x + b
        P = Φ * P * Φ' + Q

        ## Update step
        y = observations[t]
        K = P * H' / (H * P * H' + R)
        x = x + K * (y - H * x)
        P = P - K * H * P

        x_filts[t] = x
        P_filts[t] = P
    end

    return x_filts, P_filts
end

# In this specific case, however, since our model is homogenous, we do not expect to have
# any extras passed in. For convenience, we create a method without the `extra0` and
# `extras` argument which then replaces them with `nothing` and a vector of `nothing`s,
# respectively. This pattern is not specific to linear Gaussian models or Kalman filters so
# we define it with general types.

function AbstractMCMC.sample(
    model::StateSpaceModel, algorithm, observations::AbstractVector
)
    extra0 = nothing
    extras = fill(nothing, length(observations))
    return sample(model, algorithm, observations, extra0, extras)
end

# ## Simulation and Filtering

# We define the parameters for our model as so.

SEED = 1;
T = 100;
z = [-1.0, 1.0];
P = Matrix(1.0I, 2, 2);
Φ = [0.8 0.2; -0.1 0.8];
b = zeros(2);
Q = [0.2 0.0; 0.0 0.5];
H = [1.0 0.0;];
R = Matrix(0.3I, 1, 1);

# We can then construct the latent dynamics and observation process, before combining these
# to form a state space model.

dyn = LinearGaussianLatentDynamics(z, P, Φ, b, Q);
obs = LinearGaussianObservationProcess(H, R);
model = StateSpaceModel(dyn, obs);

# Synthetic data can be generated by directly sampling from the model. This calls a utility
# function from the `SSMProblems` package, which in turn, calls the three distribution
# functions we defined above.

rng = MersenneTwister(SEED);
x0, xs, ys = sample(rng, model, T);

# We can then run the Kalman filter and plot the filtering results against the ground truth.

x_filts, P_filts = AbstractMCMC.sample(model, KalmanFilter(), ys);

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
