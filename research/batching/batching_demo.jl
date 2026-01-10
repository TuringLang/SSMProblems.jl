using GeneralisedFilters

using Distributions
using LinearAlgebra
using Base.Broadcast: broadcasted
using PDMats
using StructArrays
using BenchmarkTools

using CUDA
using Magma
using Magma.LibMagma

Magma.magma_init()

# =============================================================================
# Configuration
# =============================================================================

D_state = 2
D_obs = 2
N = 3

BATCHED_CACHE_VERBOSITY[] = :debug

function kalman_predict(state, dyn_params)
    A = dyn_params[1]
    b = dyn_params[2]
    Q = dyn_params[3]

    μ̂ = A * state.μ + b
    Σ̂ = X_A_Xt(state.Σ, A) + Q
    return MvNormal(μ̂, Σ̂)
end

I_mat = CuArray{Float32}(I, D_state, D_state)
Is = SharedCuMatrix(I_mat)

μs = BatchedCuVector(CUDA.randn(Float32, D_state, N))
Σs_root = BatchedCuMatrix(CUDA.randn(Float32, D_state, D_state, N))
Σs = Σs_root .* adjoint.(Σs_root) .+ Is

As = SharedCuMatrix(CUDA.randn(Float32, D_state, D_state))
bs = BatchedCuVector(CUDA.randn(Float32, D_state, N))
Q_root = CUDA.randn(Float32, D_state, D_state)
Q = Q_root * Q_root' + I
Qs = SharedCuMatrix(Q)

Σ_PDs = broadcasted(PDMat, Σs);
Gs = MvNormal.(μs, Σ_PDs);

function kalman_predict(state, dyn_params)
    A = dyn_params[1]
    b = dyn_params[2]
    Q = dyn_params[3]

    μ̂ = A * state.μ + b
    Σ̂ = PDMat(X_A_Xt(state.Σ, A) + Q)

    return MvNormal(μ̂, Σ̂)
end

dyn_params = (As, bs, Qs)
pred_Gs = kalman_predict.(Gs, Ref(dyn_params));

# Compare to CPU
μ_test = Array(μs[end])
Σ_test = Array(Σs[end])
A_test = Array(As.data)
b_test = Array(bs[end])
Q_test = Array(Qs.data)
pred_G_test = kalman_predict(MvNormal(μ_test, PDMat(Σ_test)), (A_test, b_test, Q_test))

println("=== Predict Comparison ===\n")
println("CPU Mean: ", pred_G_test.μ)
println("GPU Mean: ", Array(pred_Gs.μ[end]))

println("CPU Covariance: ", Matrix(pred_G_test.Σ))
println("GPU Covariance: ", Array(pred_Gs.Σ.mat[end]))

# =============================================================================
# Kalman Update
# =============================================================================

function kalman_update(state, obs_params, observation)
    μ = state.μ
    Σ = state.Σ
    H = obs_params[1]
    c = obs_params[2]
    R = obs_params[3]

    # Compute innovation distribution
    m = H * μ + c
    S = PDMat(X_A_Xt(Σ, H) + R)
    ȳ = observation - m

    # Kalman gain
    K = Σ * H' / S

    # Update parameters using Joseph form for numerical stability
    μ̂ = μ + K * ȳ
    Σ̂ = PDMat(X_A_Xt(Σ, I - K * H) + X_A_Xt(R, K))

    return MvNormal(μ̂, Σ̂)
end

function kalman_step(state, dyn_params, obs_params, observation)
    state = kalman_predict(state, dyn_params)
    state = kalman_update(state, obs_params, observation)
    return state
end

# Observation parameters (H and c shared, R batched)
Hs = SharedCuMatrix(CUDA.randn(Float32, D_obs, D_state))
cs = SharedCuVector(CUDA.randn(Float32, D_obs))
I_obs = CuArray{Float32}(I, D_obs, D_obs)
I_obs_shared = SharedCuMatrix(I_obs)
Rs_root = BatchedCuMatrix(CUDA.randn(Float32, D_obs, D_obs, N))
Rs = Rs_root .* adjoint.(Rs_root) .+ I_obs_shared
Rs = PDMat.(Rs);

obs_params = (Hs, cs, Rs)

# Observations
observations = BatchedCuVector(CUDA.randn(Float32, D_obs, N))

# Run update on GPU
update_Gs = kalman_update.(pred_Gs, Ref(obs_params), observations);

# Compare update to CPU
H_test = Array(Hs.data)
c_test = Array(cs.data)
R_test = PDMat(Array(Rs.mat[end]))
obs_test = Array(observations[end])

update_G_test = kalman_update(pred_G_test, (H_test, c_test, R_test), obs_test)

println("\n=== Update Comparison ===\n")
println("CPU Mean: ", update_G_test.μ)
println("GPU Mean: ", Array(update_Gs.μ[end]))

println("CPU Covariance: ", Matrix(update_G_test.Σ))
println("GPU Covariance: ", Array(update_Gs.Σ.mat[end]))

# =============================================================================
# Full Kalman Step
# =============================================================================

# Run full step on GPU (from original state)
step_Gs = kalman_step.(Gs, Ref(dyn_params), Ref(obs_params), observations);

# Compare full step to CPU
step_G_test = kalman_step(
    MvNormal(μ_test, PDMat(Σ_test)),
    (A_test, b_test, Q_test),
    (H_test, c_test, R_test),
    obs_test,
)

println("\n=== Full Step Comparison ===\n")
println("CPU Mean: ", step_G_test.μ)
println("GPU Mean: ", Array(step_Gs.μ[end]))

println("CPU Covariance: ", Matrix(step_G_test.Σ))
println("GPU Covariance: ", Array(step_Gs.Σ.mat[end]))

# =============================================================================
# Benchmarking
# =============================================================================

using BenchmarkTools
using StaticArrays

D_bench = 10
N_bench = 60000

println("\n=== Benchmarking batched Kalman step ===\n")

println("D = $D_bench, N = $N_bench")
println(
    "Size of batched covariance matrices: ",
    round(Int, D_bench * D_bench * N_bench * sizeof(Float32) / 1024^2),
    " MB",
)

Is_bench = SharedCuMatrix(CuArray{Float32}(I, D_bench, D_bench))
μs_bench = BatchedCuVector(CUDA.randn(Float32, D_bench, N_bench))
Σs_root_bench = BatchedCuMatrix(CUDA.randn(Float32, D_bench, D_bench, N_bench))
Σs_bench = Σs_root_bench .* adjoint.(Σs_root_bench) .+ Is_bench
Σ_PDs_bench = broadcasted(PDMat, Σs_bench);
Gs_bench = MvNormal.(μs_bench, Σ_PDs_bench);

As_bench = SharedCuMatrix(CUDA.randn(Float32, D_bench, D_bench))
bs_bench = BatchedCuVector(CUDA.randn(Float32, D_bench, N_bench))
Qs_root_bench = CUDA.randn(Float32, D_bench, D_bench)
Qs_bench_mat = Qs_root_bench * adjoint(Qs_root_bench)
Qs_bench_mat += I
Qs_bench = SharedCuMatrix(Qs_bench_mat)
dyn_params_bench = (As_bench, bs_bench, Qs_bench)

Hs_bench = SharedCuMatrix(CUDA.randn(Float32, D_bench, D_bench))
cs_bench = SharedCuVector(CUDA.randn(Float32, D_bench))
Rs_root_bench = BatchedCuMatrix(CUDA.randn(Float32, D_bench, D_bench, N_bench))
Rs_bench = Rs_root_bench .* adjoint.(Rs_root_bench) .+ Is_bench
Rs_bench = PDMat.(Rs_bench)
obs_params_bench = (Hs_bench, cs_bench, Rs_bench)

ys_bench = BatchedCuVector(CUDA.randn(Float32, D_bench, N_bench))

println("\nBenchmarking batched Kalman step...\n")

BATCHED_CACHE_VERBOSITY[] = :silent
display(
    @benchmark kalman_step.(
        $Gs_bench, Ref($dyn_params_bench), Ref($obs_params_bench), $ys_bench
    )
)

# Compare to static arrays
μs_static = [@SVector randn(Float32, D_bench) for _ in 1:N_bench]
Σs_static = [
    begin
        A = @SMatrix randn(Float32, D_bench, D_bench)
        A * A' + I
    end for _ in 1:N_bench
]
Gs_static = [MvNormal(μs_static[n], PDMat(Σs_static[n])) for n in 1:N_bench]

As_static = @SMatrix randn(Float32, D_bench, D_bench)
bs_static = [@SVector randn(Float32, D_bench) for _ in 1:N_bench]
Qs_root_static = @SMatrix randn(Float32, D_bench, D_bench)
Qs_static_mat = Qs_root_static * adjoint(Qs_root_static) + I
Qs_static = Qs_static_mat
dyn_params_static = (As_static, bs_static, Qs_static)

Hs_static = @SMatrix randn(Float32, D_bench, D_bench)
cs_static = @SVector randn(Float32, D_bench)
Rs_root_static = [@SMatrix randn(Float32, D_bench, D_bench) for _ in 1:N_bench]
Rs_static = [
    Symmetric(R_root * R_root') + SMatrix{D_bench,D_bench,Float32}(I) for
    R_root in Rs_root_static
]
obs_static = [@SVector randn(Float32, D_bench) for _ in 1:N_bench]
obs_params_static = (Hs_static, cs_static, PDMat.(Rs_static))

function test_static(Gs, dyn_params, obs_params, observations, out)
    N = length(out)
    Threads.@threads for n in 1:N
        @inbounds out[n] = kalman_step(
            Gs[n],
            (dyn_params[1], dyn_params[2][n], dyn_params[3]),
            (obs_params[1], obs_params[2], obs_params[3][n]),
            observations[n],
        )
    end
    return out
end

out = Vector{eltype(Gs_static)}(undef, N_bench)

println("\nBenchmarking static Kalman step...\n")

test_static(Gs_static, dyn_params_static, obs_params_static, obs_static, out);  # warm-up

display(
    @benchmark test_static(
        $Gs_static, $dyn_params_static, $obs_params_static, $obs_static, $out
    )
)

println("\nBenchmarking single batched matrix multiplication...\n")

display(@benchmark $Σs_root_bench .* $Σs_root_bench)

tot_mem = 3 * D_bench * D_bench * N_bench * sizeof(Float32)
throughput = 1008 * 10^9
println("\nTheoretical optimum time: ", tot_mem / throughput * 10^6, " μs")
