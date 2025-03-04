using GeneralisedFilters
using Distributions
using HypothesisTests
using LinearAlgebra
using LogExpFunctions: softmax
using StableRNGs
using StatsBase

using CUDA
using NNlib

D_outer = 2
D_inner = 3
D_obs = 2
N_particles = 100000
T = 10

# Define inner dynamics
struct InnerDynamics{T} <: LinearGaussianLatentDynamics{T}
    μ0::Vector{T}
    Σ0::Matrix{T}
    A::Matrix{T}
    b::Vector{T}
    C::Matrix{T}
    Q::Matrix{T}
end

# CPU methods
GeneralisedFilters.calc_μ0(dyn::InnerDynamics; kwargs...) = dyn.μ0
GeneralisedFilters.calc_Σ0(dyn::InnerDynamics; kwargs...) = dyn.Σ0
GeneralisedFilters.calc_A(dyn::InnerDynamics, ::Integer; kwargs...) = dyn.A
function GeneralisedFilters.calc_b(dyn::InnerDynamics, ::Integer; prev_outer, kwargs...)
    return dyn.b + dyn.C * prev_outer
end
GeneralisedFilters.calc_Q(dyn::InnerDynamics, ::Integer; kwargs...) = dyn.Q

# GPU methods
function GeneralisedFilters.batch_calc_μ0s(dyn::InnerDynamics{T}, N; kwargs...) where {T}
    μ0s = CuArray{T}(undef, length(dyn.μ0), N)
    return μ0s[:, :] .= cu(dyn.μ0)
end
function GeneralisedFilters.batch_calc_Σ0s(
    dyn::InnerDynamics{T}, N::Integer; kwargs...
) where {T}
    Σ0s = CuArray{T}(undef, size(dyn.Σ0)..., N)
    return Σ0s[:, :, :] .= cu(dyn.Σ0)
end
function GeneralisedFilters.batch_calc_As(
    dyn::InnerDynamics{T}, ::Integer, N::Integer; kwargs...
) where {T}
    As = CuArray{T}(undef, size(dyn.A)..., N)
    As[:, :, :] .= cu(dyn.A)
    return As
end
function GeneralisedFilters.batch_calc_bs(
    dyn::InnerDynamics{T}, ::Integer, N::Integer; prev_outer, kwargs...
) where {T}
    Cs = CuArray{T}(undef, size(dyn.C)..., N)
    Cs[:, :, :] .= cu(dyn.C)
    return NNlib.batched_vec(Cs, prev_outer) .+ cu(dyn.b)
end
function GeneralisedFilters.batch_calc_Qs(
    dyn::InnerDynamics{T}, ::Integer, N::Integer; kwargs...
) where {T}
    Q = CuArray{T}(undef, size(dyn.Q)..., N)
    return Q[:, :, :] .= cu(dyn.Q)
end

rng = StableRNG(1234)
μ0 = rand(rng, Float32, D_outer + D_inner)
Σ0s = [rand(rng, Float32, D_outer, D_outer), rand(rng, Float32, D_inner, D_inner)]
Σ0s = [Σ * Σ' for Σ in Σ0s]  # make Σ0 positive definite
Σ0 = [
    Σ0s[1] zeros(Float32, D_outer, D_inner)
    zeros(Float32, D_inner, D_outer) Σ0s[2]
]
A = [
    rand(rng, Float32, D_outer, D_outer) zeros(Float32, D_outer, D_inner)
    rand(rng, Float32, D_inner, D_outer + D_inner)
]
# Make mean-reverting
A /= 3.0f0
A[diagind(A)] .= -0.5f0
b = rand(rng, Float32, D_outer + D_inner)
Qs = [rand(rng, Float32, D_outer, D_outer), rand(rng, Float32, D_inner, D_inner)] ./ 10.0f0
Qs = [Q * Q' for Q in Qs]  # make Q positive definite
Q = [
    Qs[1] zeros(Float32, D_outer, D_inner)
    zeros(Float32, D_inner, D_outer) Qs[2]
]
H = [zeros(Float32, D_obs, D_outer) rand(rng, Float32, D_obs, D_inner)]
c = rand(rng, Float32, D_obs)
R = rand(rng, Float32, D_obs, D_obs)
R = R * R' / 3.0f0  # make R positive definite

observations = [rand(rng, Float32, D_obs) for _ in 1:T]

# Rao-Blackwellised particle filtering

outer_dyn = GeneralisedFilters.HomogeneousLinearGaussianLatentDynamics(
    μ0[1:D_outer], Σ0[1:D_outer, 1:D_outer], A[1:D_outer, 1:D_outer], b[1:D_outer], Qs[1]
)
inner_dyn = InnerDynamics(
    μ0[(D_outer + 1):end],
    Σ0[(D_outer + 1):end, (D_outer + 1):end],
    A[(D_outer + 1):end, (D_outer + 1):end],
    b[(D_outer + 1):end],
    A[(D_outer + 1):end, 1:D_outer],
    Qs[2],
)
obs = GeneralisedFilters.HomogeneousLinearGaussianObservationProcess(
    H[:, (D_outer + 1):end], c, R
)
hier_model = HierarchicalSSM(outer_dyn, inner_dyn, obs)

cpu_rbpf = RBPF(
    KalmanFilter(), N_particles; threshold=0.8, resampler=GeneralisedFilters.Multinomial()
)
states, ll = GeneralisedFilters.filter(rng, hier_model, cpu_rbpf, observations)

gpu_rbpf = BatchRBPF(
    BatchKalmanFilter(N_particles),
    N_particles;
    threshold=0.8,
    resampler=GeneralisedFilters.Multinomial(),
)
states, ll = GeneralisedFilters.filter(hier_model, gpu_rbpf, observations)

# Benchmarking
using BenchmarkTools

@benchmark GeneralisedFilters.filter(rng, hier_model, cpu_rbpf, observations)
@benchmark CUDA.@sync GeneralisedFilters.filter(hier_model, gpu_rbpf, observations)

# Profiling

CUDA.@profile GeneralisedFilters.filter(hier_model, gpu_rbpf, observations)

# Test for multiple values of N_particles
# Ns = [10, 100, 1000, 10000, 100000, 10^6]
Ns = 10 .^ range(1, 5; step=0.5)
Ns = round.(Int, Ns)
cpu_results = []
gpu_results = []

for N in Ns
    println(N)
    cpu_rbpf = RBPF(
        KalmanFilter(), N; threshold=0.8, resampler=GeneralisedFilters.Multinomial()
    )
    gpu_rbpf = BatchRBPF(
        BatchKalmanFilter(N), N; threshold=0.8, resampler=GeneralisedFilters.Multinomial()
    )

    cpu_res = @benchmark GeneralisedFilters.filter(rng, hier_model, cpu_rbpf, observations) seconds =
        20
    gpu_res = @benchmark CUDA.@sync GeneralisedFilters.filter(
        hier_model, gpu_rbpf, observations
    )

    push!(cpu_results, cpu_res)
    push!(gpu_results, gpu_res)
end

# Save results
using JLD2

jldopen("cpu_results.jld2", "w") do file
    file["cpu_results"] = cpu_results
end

jldopen("gpu_results.jld2", "w") do file
    file["gpu_results"] = gpu_results
end

# Load results

cpu_results = jldopen("cpu_results.jld2") do file
    file["cpu_results"]
end
gpu_results = jldopen("gpu_results.jld2") do file
    file["gpu_results"]
end

# Plot mean times with error bars
using Plots

p = plot(;
    xlabel="Number of Particles",
    ylabel="Mean Wall Time Per Step ± 1 SE (ms)",
    xticks=10 .^ range(1, 5; step=1.0),
    y_ticks=[0.1, 1, 10, 100, 100],
    yscale=:log10,
    xscale=:log10,
    legend=:topleft,
)

cpu_means = [mean(cpu_res.times / 10^7) for cpu_res in cpu_results]
gpu_means = [mean(gpu_res.times / 10^7) for gpu_res in gpu_results]

cpu_stds = [std(cpu_res.times / 10^7) for cpu_res in cpu_results]
gpu_stds = [std(gpu_res.times / 10^7) for gpu_res in gpu_results]

cpu_std_error = cpu_stds ./ sqrt.(length.(cpu_results)) * 1.96
gpu_std_error = gpu_stds ./ sqrt.(length.(gpu_results)) * 1.96

plot!(p, Ns, cpu_means; ribbon=cpu_std_error, label="CPU Implementation")
plot!(p, Ns, gpu_means; ribbon=gpu_std_error, label="GPU Implementation")
