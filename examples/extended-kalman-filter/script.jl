# # Extended Kalman filter for a non-linear SSM: sine signal
using GaussianDistributions: correct, Gaussian
using LinearAlgebra
using Statistics
using Plots
using Random
using ForwardDiff: jacobian
using SSMProblems

# Model definition
struct SineModel <: AbstractStateSpaceModel
    """
    """
end


f(x::AbstractArray, dt::Float64) = [x[1] + x[2] * dt; x[2] - sin(x[1])*dt]
jacob(x) = jacobian(state -> f(state, dt), x)

f0(model::RangeBearingTracking) = Gaussian(model.z, model.P)
f(x::Vector{Float64}, model::RangeBearingTracking) = Gaussian(model.Φ * x + model.b, model.Q)
g(y::Vector{Float64}, model::RangeBearingTracking) = Gaussian(model.H * y, model.R)

function transition!!(rng::AbstractRNG, model::RangeBearingTracking)
    return Gaussian(model.z, model.P)
end

function transition!!(rng::AbstractRNG, model::RangeBearingTracking, state::Gaussian)
    let Φ = model.Φ, Q = model.Q, μ = state.μ, Σ = state.Σ
        return Gaussian(Φ * μ, Φ * Σ * Φ' + Q)
    end
end

# Simulation parameters
SEED = 1
T = 5
nstep = 100
dt = T / nstep
Q = [
  dt^3/3 dt^2/2;
  dt^2/2 dt
]
R = 1.0*I

model = NonLinearSSM()

# Generate synthetic data
rng = MersenneTwister(SEED)
x, y = Vector{Any}(undef, T), Vector{Any}(undef, T)
x[1] = rand(rng, f0(model))
for t in 1:T
    y[t] = rand(rng, g(x[t], model))
    if t < T
        x[t + 1] = rand(rng, f(x[t], model))
    end
end

# Kalman filter
function filter(rng::Random.AbstractRNG, model::RangeBearingTracking, y::Vector{Any})
    T = length(y)
    p = transition!!(rng, model)
    ps = [p]
    for i in 1:T
        p = transition!!(rng, model, p)
        p, yres, _ = correct(p, Gaussian(y[i], model.R), model.H)
        push!(ps, p)
    end
    return ps
end

# Run filter and plot results
ps = filter(rng, model, y)

p_mean = mean.(ps)
p_cov = sqrt.(cov.(ps))

p1 = scatter(1:T, first.(y); color="red", label="Observations")
plot!(
    p1,
    0:T,
    first.(p_mean);
    color="orange",
    label="Filtered x1",
    grid=false,
    ribbon=getindex.(p_cov, 1, 1),
    fillalpha=0.5,
)

plot!(
    p1,
    0:T,
    last.(p_mean);
    color="blue",
    label="Filtered x2",
    grid=false,
    ribbon=getindex.(p_cov, 2, 2),
    fillalpha=0.5,
)
