# # Kalman filter using Kalman.jl
using GaussianDistributions: correct, Gaussian
using LinearAlgebra
using Statistics
using Plots
using Random
using SSMProblems

# Model definition
struct LinearGaussianSSM <: AbstractStateSpaceModel
    """
        A state space model with linear dynamics and Gaussian noise.
        The model is defined by the following equations:
        x[0] = z + ϵ,                 ϵ    ∼ N(0, P)
        x[k] = Φx[k-1] + b + w[k],    w[k] ∼ N(0, Q)
        y[k] = Hx[k] + v[k],          v[k] ∼ N(0, R)
    """
    z::Vector{Float64}
    P::Matrix{Float64}
    Φ::Matrix{Float64}
    b::Vector{Float64}
    Q::Matrix{Float64}
    H::Matrix{Float64}
    R::Matrix{Float64}
end

f0(model::LinearGaussianSSM) = Gaussian(model.z, model.P)
f(x::Vector{Float64}, model::LinearGaussianSSM) = Gaussian(model.Φ * x + model.b, model.Q)
g(y::Vector{Float64}, model::LinearGaussianSSM) = Gaussian(model.H * y, model.R)

function transition!!(rng::AbstractRNG, model::LinearGaussianSSM)
    return Gaussian(model.z, model.P)
end

function transition!!(rng::AbstractRNG, model::LinearGaussianSSM, state::Gaussian)
    let Φ = model.Φ, Q = model.Q, μ = state.μ, Σ = state.Σ
        return Gaussian(Φ * μ, Φ * Σ * Φ' + Q)
    end
end

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

model = LinearGaussianSSM(z, P, Φ, b, Q, H, R)

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
function filter(rng::Random.AbstractRNG, model::LinearGaussianSSM, y::Vector{Any})
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
