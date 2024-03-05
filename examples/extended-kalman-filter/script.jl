# # Extended Kalman filter for a non-linear SSM: sine signal
using GaussianDistributions: correct, Gaussian
using LinearAlgebra
using Statistics
using Plots
using Random
using ForwardDiff: jacobian
using SSMProblems

struct PendulumModel
    x0::Vector{Float64}
    dt::Float64

    Q::AbstractMatrix
    R::AbstractMatrix
end

# Simulation parameters
SEED = 4
T = 5.0
dt = 0.0125
nstep = Int(T / dt)
g = 9.8
r = 0.3
qc = 1.0

x0 = [pi / 2; 0]
Q = qc .* [
    dt^3/3 dt^2/2
    dt^2/2 dt
]
model = PendulumModel(x0, dt, Q, r^2 * I(1))

f(x::Array, model::PendulumModel) =
    let dt = model.dt
        [x[1] + x[2] * dt, x[2] - g * sin(x[1]) * dt]
    end
h(x::Array, model::PendulumModel) = [sin(x[1])]

function transition!!(::AbstractRNG, model::PendulumModel)
    return Gaussian(model.x0, zeros(2, 2))
end

function transition!!(::AbstractRNG, model::PendulumModel, state::Gaussian)
    Jf = jacobian(x -> f(x, model), state.μ)
    Jh = jacobian(x -> h(x, model), state.μ)
    pred = f(state.μ, model)
    return Gaussian(pred, Jf * state.Σ * Jf' + model.Q)
end

# Generate synthetic data
rng = MersenneTwister(SEED)
x, y = Vector{Any}(undef, nstep), Vector{Any}(undef, nstep)
x[1] = x0
for t in 1:nstep
    y[t] = rand(rng, Gaussian(h(x[t], model), model.R))
    if t < nstep
        x[t + 1] = rand(rng, Gaussian(f(x[t], model), model.Q))
    end
end

function ekf_correct(obs, state::Gaussian, model::PendulumModel)
    Jf = jacobian(x -> f(x, model), state.μ)
    Jh = jacobian(x -> h(x, model), state.μ)

    S = model.R + Jh * state.Σ * Jh'
    K = state.Σ * Jh' / S
    pred = state.μ + K * (obs - h(state.μ, model))
    return Gaussian(pred, (I - K * Jh) * state.Σ)
end

# Extended Kalman filter
function filter(rng::Random.AbstractRNG, model::PendulumModel, y::Vector)
    T = length(y)
    p = transition!!(rng, model)
    ps = [p]
    for i in 2:T
        p = transition!!(rng, model, p)
        p = ekf_correct(y[i], p, model)
        push!(ps, p)
    end
    return ps
end

ps = filter(rng, model, y)
ts = dt:dt:T
filtered_mean = first.(mean.(ps))

plot(ts, first.(x); color=:gray, label="Latent state")

scatter!(
    ts, first.(y); markersize=1, markerstrokealpha=0, label="Observations", color=:black
)

plot!(ts, filtered_mean; label="Filtered mean", color=:red)
