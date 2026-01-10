using GeneralisedFilters
const GF = GeneralisedFilters

using Distributions
using LinearAlgebra
using Base.Broadcast: broadcasted
using PDMats
using StructArrays
using BenchmarkTools
import Distributions: params

using CUDA
using Magma
using Magma.LibMagma

Magma.magma_init()
BATCHED_CACHE_VERBOSITY[] = :debug

D_state = 3
D_obs = 2
N = 3

μs = BatchedCuVector(CUDA.randn(Float32, D_state, N))
Σs_root = BatchedCuMatrix(CUDA.randn(Float32, D_state, D_state, N))
Σs = Σs_root .* adjoint.(Σs_root) .+ Ref(I)

As = BatchedCuMatrix(CUDA.randn(Float32, D_state, D_state, N))
bs = BatchedCuVector(CUDA.randn(Float32, D_state, N))
Q_root = BatchedCuMatrix(CUDA.randn(Float32, D_state, D_state, N))
Qs = Q_root .* adjoint.(Q_root) .+ Ref(I)
Qs = PDMat.(Qs);

Σ_PDs = broadcasted(PDMat, Σs);
Gs = MvNormal.(μs, Σ_PDs);

# Observation parameters (H and c shared, R batched)
Hs = BatchedCuMatrix(CUDA.randn(Float32, D_obs, D_state, N))
cs = BatchedCuVector(CUDA.randn(Float32, D_obs, N))
Rs_root = BatchedCuMatrix(CUDA.randn(Float32, D_obs, D_obs, N))
Rs = Rs_root .* adjoint.(Rs_root) .+ Ref(I)
Rs = PDMat.(Rs);

# Observations
observations = BatchedCuVector(CUDA.randn(Float32, D_obs, N))
jitter = 1f-6

dyn_params = tuple.(As, bs, Qs)
obs_params = tuple.(Hs, cs, Rs)

# Dispatch-based jitter application (avoids if statements in batched code)
_maybe_apply_jitter(Σ, ::Nothing) = Σ
_maybe_apply_jitter(Σ, jitter::Real) = Σ + jitter * I

function kalman_step(state, dyn_params, obs_params, observation, jitter)
    μ, Σ = params(state)
    A, b, Q = dyn_params

    μ = A * μ + b
    Σ = X_A_Xt(Σ, A) + Q

    H, c, R = obs_params

    z = GF._compute_innovation(μ, H, c, observation)
    S = GF._compute_innovation_cov(Σ, H, R)
    K = GF._compute_kalman_gain(Σ, H, S)
    _, Σ̂_raw = GF._compute_joseph_update(Σ, K, H, R)

    μ = μ + K * z
    Σ = PDMat(_maybe_apply_jitter(Σ̂_raw, jitter))

    # ll = logpdf(MvNormal(z, S), zero(z))

    return MvNormal(μ, Σ)
end

res = kalman_step.(Gs, dyn_params, obs_params, observations, Ref(jitter))
println(typeof(res))
println(eltype(res))
