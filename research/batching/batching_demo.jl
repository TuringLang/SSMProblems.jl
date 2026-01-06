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

println("CPU Mean: ", pred_G_test.μ[1:5])
println("GPU Mean: ", Array(pred_Gs[end].μ[1:5]))

println("CPU Covariance Diagonal: ", diag(pred_G_test.Σ)[1:5])
println("GPU Covariance Diagonal: ", Array(diag(pred_Gs[end].Σ))[1:5])

# Increase batch size and benchmark
D_large = 32
N_large = 10000
μs_large = BatchedCuVector(CUDA.randn(Float32, D_large, N_large))
Σs_root_large = BatchedCuMatrix(CUDA.randn(Float32, D_large, D_large, N_large))
Σs_large = Σs_root_large .* adjoint.(Σs_root_large) .+ SharedCuMatrix(CuArray{Float32}(I, D_large, D_large))
Σ_PDs_large = broadcasted(PDMat, Σs_large);
Gs_large = StructArray{MvNormal}((μ=μs_large, Σ=Σ_PDs_large));
dyn_params_large = (
    SharedCuMatrix(CUDA.randn(Float32, D_large, D_large)),
    BatchedCuVector(CUDA.randn(Float32, D_large, N_large)),
    SharedCuMatrix((CUDA.randn(Float32, D_large, D_large) * CUDA.randn(Float32, D_large, D_large)') .+ CuArray{Float32}(I, D_large, D_large)),
)
display(@benchmark kalman_predict.($Gs_large, Ref($dyn_params_large)))

# Compare to multithreading StaticArrays
using StaticArrays
μs_static = [SVector{D_large, Float32}(randn(Float32, D_large)) for _ in 1:N_large];
Σs_root_static = [SMatrix{D_large,D_large,Float32}(randn(Float32, D_large, D_large)) for _ in 1:N_large];
Σs_static = [Σs_root_static[i] * adjoint(Σs_root_static[i]) + I for i in 1:N_large];
Gs_static = [MvNormal(μs_static[i], Σs_static[i]) for i in 1:N_large];
A_static = SMatrix{D_large,D_large,Float32}(randn(Float32, D_large, D_large));
b_static = [SVector{D_large, Float32}(randn(Float32, D_large)) for _ in 1:N_large];
Q_root_static = SMatrix{D_large,D_large,Float32}(randn(Float32, D_large, D_large));
Q_static = Q_root_static * adjoint(Q_root_static) + I;

function test_static(Gs, A, b, Q)
    out = Vector{MvNormal{Float32, PDMat{Float32, SMatrix{32, 32, Float32, 1024}}, SVector{32, Float32}}}(undef, length(Gs))
    for i in 1:length(Gs)
        @inbounds out[i] = kalman_predict(Gs[i], (A, b[i], Q))
    end
    return out
end

display(@benchmark test_static($Gs_static, $A_static, $b_static, $Q_static))

@profview test_static(Gs_static, A_static, b_static, Q_static)
