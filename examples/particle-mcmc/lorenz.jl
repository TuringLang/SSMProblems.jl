using Distributions
using Random
using SSMProblems
using UnPack
using OrdinaryDiffEq
using LinearAlgebra
using PDMats
using GLMakie

include("particles.jl")
include("resamplers.jl")
include("simple-filters.jl")

Base.@kwdef struct Parameters{T<:Real}
    β::T = 8 / 3
    ρ::T = 28.0
    σ::T = 10.0
    ν::T = 1.0 # Obs noise variance
    dt::T = 0.025 # Time step
end

function lorenz!(du, u, p::Parameters, t)
    @unpack β, ρ, σ = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    return du[3] = u[1] * u[2] - β * u[3]
end

struct LatentNoiseProcess{T} <: LatentDynamics{Vector{T}}
    σ::AbstractPDMat{T}
    dt::T
    integrator
end

struct ObservationNoiseProcess{T} <: ObservationProcess{Vector{T}}
    σ::AbstractPDMat{T}
end

function SSMProblems.distribution(dyn::LatentNoiseProcess, step::Integer, prev_state, extra)
    reinit!(dyn.integrator, prev_state)
    step!(dyn.integrator, dyn.dt, true)
    return MvNormal(dyn.integrator.u, dyn.σ)
end

function SSMProblems.distribution(dyn::LatentNoiseProcess, extra)
    return MvNormal([1; 0; 0], dyn.σ)
end

function SSMProblems.distribution(obs::ObservationNoiseProcess, step::Integer, state, extra)
    return MvNormal(state, obs.σ * I)
end

# Simulate some data
u0 = [1.0; 0.0; 0.0]
params = Parameters()

dt = 0.025
N = 100
Np = 512
tspan = (0.0, dt * N)

rng = MersenneTwister()

prob = ODEProblem(lorenz!, u0, tspan, params)
alg = Tsit5()
integrator = init(prob, Tsit5(); dt=dt, adaptive=false)
sol = solve(prob, alg; dt=dt, adaptive=false)

# SSM Noise Model
dyn = LatentNoiseProcess(ScalMat(3, params.dt), params.dt, integrator)
obs = ObservationNoiseProcess(ScalMat(3, params.ν))
model = StateSpaceModel(dyn, obs)
x0, x, y = sample(rng, model, N)

filter = BF(Np; threshold=1.0, resampler=Systematic());
sparse_ancestry = AncestorCallback(eltype(model.dyn), filter.N, 1.0);
tree, llbf = sample(rng, model, filter, y; callback=sparse_ancestry);
lineage = get_ancestry(sparse_ancestry.tree)

# Fancy 3D plot
# fig = Figure()
# lines(fig[1, 1], hcat(x0, x...))
# for (i, path) in enumerate(lineage)
# 	lines!(fig[1, 1], hcat(path...), color=:black)
# end

fig = Figure()
for i in eachindex(first(x))
    lines(fig[i, 1], hcat(x0, x...)[i, :])
    for path in lineage
        lines!(fig[i, 1], hcat(path...)[i, :]; color=:black, alpha=0.1)
    end
end
