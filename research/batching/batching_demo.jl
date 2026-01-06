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

D_state = 64
D_obs = 64
N = 1000

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
Gs = StructArray{MvNormal}((μ=μs, Σ=Σ_PDs));

function kalman_predict(state, dyn_params)
    A = dyn_params[1]
    b = dyn_params[2]
    Q = dyn_params[3]

    μ̂ = A * state.μ + b
    Σ̂ = X_A_Xt(state.Σ, A) + Q
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

println("=== Predict Comparison ===")
println("CPU Mean: ", pred_G_test.μ[1:5])
println("GPU Mean: ", Array(pred_Gs[end].μ[1:5]))

println("CPU Covariance [1:3, 1:3]: ", Matrix(pred_G_test.Σ)[1:3, 1:3])
println("GPU Covariance [1:3, 1:3]: ", Array(pred_Gs[end].Σ)[1:3, 1:3])

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
    Σ̂ = X_A_Xt(Σ, I - K * H) + X_A_Xt(R, K)

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

obs_params = (Hs, cs, Rs)

# Observations
observations = BatchedCuVector(CUDA.randn(Float32, D_obs, N))

# Run update on GPU
update_Gs = kalman_update.(pred_Gs, Ref(obs_params), observations);

# Compare update to CPU
H_test = Array(Hs.data)
c_test = Array(cs.data)
R_test = PDMat(Array(Rs[end]))
obs_test = Array(observations[end])

update_G_test = kalman_update(pred_G_test, (H_test, c_test, R_test), obs_test)

println("\n=== Update Comparison ===")
println("CPU Mean: ", update_G_test.μ[1:5])
println("GPU Mean: ", Array(update_Gs.μ[end][1:5]))

println("CPU Covariance [1:3, 1:3]: ", Matrix(update_G_test.Σ)[1:3, 1:3])
println("GPU Covariance [1:3, 1:3]: ", Array(update_Gs.Σ[end])[1:3, 1:3])

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

println("\n=== Full Step Comparison ===")
println("CPU Mean: ", step_G_test.μ[1:5])
println("GPU Mean: ", Array(step_Gs.μ[end][1:5]))

println("CPU Covariance [1:3, 1:3]: ", Matrix(step_G_test.Σ)[1:3, 1:3])
println("GPU Covariance [1:3, 1:3]: ", Array(step_Gs.Σ[end])[1:3, 1:3])
