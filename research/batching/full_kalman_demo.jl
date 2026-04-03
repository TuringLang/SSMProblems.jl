using GeneralisedFilters
const GF = GeneralisedFilters

using Distributions
using LinearAlgebra
using Base.Broadcast: broadcasted
using PDMats
using BenchmarkTools
using StaticArrays
import Distributions: params

using CUDA
using Magma
using Magma.LibMagma

Magma.magma_init()
BATCHED_CACHE_VERBOSITY[] = :debug

D_state = 10
D_obs = 10
N = 10000

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
Hs = Shared(CUDA.randn(Float32, D_obs, D_state), N)
cs = Shared(CUDA.randn(Float32, D_obs), N)
Rs_root = BatchedCuMatrix(CUDA.randn(Float32, D_obs, D_obs, N))
Rs = Rs_root .* adjoint.(Rs_root) .+ Ref(I)
Rs = PDMat.(Rs);

# Observations
observations = BatchedCuVector(CUDA.randn(Float32, D_obs, N))
jitter = 1.0f-6

dyn_params = tuple.(As, bs, Qs)
obs_params = tuple.(Hs, cs, Rs)

# Dispatch-based jitter application (avoids if statements in batched code)
_maybe_apply_jitter(Σ, ::Nothing) = Σ
_maybe_apply_jitter(Σ, jitter::Real) = Σ + jitter * I

function kalman_step(state, dyn_params, obs_params, observation, jitter)
    μ, Σ = params(state)
    A, b, Q = dyn_params

    μ = A * μ + b
    Σ = PDMat(X_A_Xt(Σ, A) + Q)

    H, c, R = obs_params

    z = GF._compute_innovation(μ, H, c, observation)
    S = GF._compute_innovation_cov(Σ, H, R)
    K = GF._compute_kalman_gain(Σ, H, S)
    _, Σ̂_raw = GF._compute_joseph_update(Σ, K, H, R)

    μ = μ + K * z
    Σ = PDMat(_maybe_apply_jitter(Σ̂_raw, jitter))

    ll = Distributions._logpdf(MvNormal(z, S), zero(z))

    return MvNormal(μ, Σ), ll
end

res = kalman_step.(Gs, dyn_params, obs_params, observations, Ref(jitter))
println("\nFull type: ", typeof(res))
println("\nElement type: ", eltype(res))

# Access second component of batched tuple (the log-likelihoods)
lls = res.components[2]

######################
#### BENCHMARKING ####
######################

BATCHED_CACHE_VERBOSITY[] = :silent

# Convert parameters to CPU StaticArrays
μs_static = [SVector{D_state}(Array(μs[i])) for i in 1:N];
Σs_static = [SMatrix{D_state,D_state}(Array(Σs[i])) for i in 1:N];
As_static = [SMatrix{D_state,D_state}(Array(As[i])) for i in 1:N];
bs_static = [SVector{D_state}(Array(bs[i])) for i in 1:N];
Qs_static = [PDMat(SMatrix{D_state,D_state}(Array(Qs.mat[i]))) for i in 1:N];
Hs_static = [SMatrix{D_obs,D_state}(Array(Hs[i])) for i in 1:N];
cs_static = [SVector{D_obs}(Array(cs[i])) for i in 1:N];
Rs_static = [PDMat(SMatrix{D_obs,D_obs}(Array(Rs.mat[i]))) for i in 1:N];
observations_static = [SVector{D_obs}(Array(observations[i])) for i in 1:N];

Gs_static = MvNormal.(μs_static, Σs_static);
dyn_params_static = tuple.(As_static, bs_static, Qs_static);
obs_params_static = tuple.(Hs_static, cs_static, Rs_static);

println("\nBenchmarking GPU batched Kalman step...")
display(@benchmark begin
    res = kalman_step.($Gs, $dyn_params, $obs_params, $observations, Ref($jitter))
    LibMagma.magma_queue_sync_internal(get_magma_queue(), C_NULL, C_NULL, 0)
end)

println("\nBenchmarking CPU StaticArrays Kalman step...")
display(
    @benchmark begin
        res_static =
            kalman_step.(
                $Gs_static,
                $dyn_params_static,
                $obs_params_static,
                $observations_static,
                Ref($jitter),
            )
    end
)
