using CUDA
using NNlib

function filter_allocations(
    μ0s::CuArray{Float32,2},
    Σ0s::CuArray{Float32,3},
    As::CuArray{Float32,3},
    bs::CuArray{Float32,2},
    Qs::CuArray{Float32,3},
    Hs::CuArray{Float32,3},
    cs::CuArray{Float32,2},
    Rs::CuArray{Float32,3},
    ys::Vector{Vector{Float32}},
)
    μs = μ0s
    Σs = Σ0s

    for y in ys
        # Predict
        μs = NNlib.batched_vec(As, μs) .+ bs
        Σs = NNlib.batched_mul(NNlib.batched_mul(As, Σs), NNlib.batched_transpose(As)) .+ Qs

        # Innovation
        m = NNlib.batched_vec(Hs, μs) .+ cs
        y_res = cu(y) .- m
        S = NNlib.batched_mul(NNlib.batched_mul(Hs, Σs), NNlib.batched_transpose(Hs)) .+ Rs

        # Kalman gain
        d_ipiv, _, d_S = CUDA.CUBLAS.getrf_strided_batched(S, true)
        ΣHT = NNlib.batched_mul(Σs, NNlib.batched_transpose(Hs))
        S_inv = CuArray{Float32}(undef, size(S))
        CUDA.CUBLAS.getri_strided_batched!(d_S, S_inv, d_ipiv)
        K = NNlib.batched_mul(ΣHT, S_inv)

        # Update
        μs = μs .+ NNlib.batched_vec(K, y_res)
        Σs = Σs .- NNlib.batched_mul(K, NNlib.batched_mul(Hs, Σs))
    end

    return μs, Σs
end

function filter_inplace(
    μ0s::CuArray{Float32,2},
    Σ0s::CuArray{Float32,3},
    As::CuArray{Float32,3},
    bs::CuArray{Float32,2},
    Qs::CuArray{Float32,3},
    Hs::CuArray{Float32,3},
    cs::CuArray{Float32,2},
    Rs::CuArray{Float32,3},
    ys::Vector{Vector{Float32}},
)
    μs = μ0s
    Σs = Σ0s

    D, N = size(μ0s)

    # Containers
    Aμs = CuArray{Float32}(undef, D, N)
    AΣs = CuArray{Float32}(undef, D, D, N)

    Hμs = CuArray{Float32}(undef, D, N)
    m = CuArray{Float32}(undef, D, N)
    S = CuArray{Float32}(undef, D, D, N)
    d_ipiv = CuArray{Int32}(undef, D, N)
    y_res = CuArray{Float32}(undef, D, N)
    ΣHT = CuArray{Float32}(undef, D, D, N)
    S_inv = CuArray{Float32}(undef, D, D, N)
    K = CuArray{Float32}(undef, D, D, N)
    HΣs = CuArray{Float32}(undef, D, D, N)

    for y in ys
        # Predict
        CUDA.CUBLAS.gemv_strided_batched!('N', 1.0f0, As, μs, 0.0f0, Aμs)
        # NNlib.batched_vec!(As, μs, Aμs)
        μs .= Aμs .+ bs

        NNlib.batched_mul!(AΣs, As, Σs)
        NNlib.batched_mul!(AΣs, NNlib.batched_transpose(As), Qs, 1.0f0, 1.0f0)
        Σs .= AΣs

        # Innovation
        CUDA.CUBLAS.gemv_strided_batched!('N', 1.0f0, Hs, μs, 0.0f0, Hμs)
        # NNlib.batched_vec!(Hμs, Hs, μs)
        m .= Hμs .+ cs

        NNlib.batched_mul!(ΣHT, Σs, NNlib.batched_transpose(Hs))
        NNlib.batched_mul!(S, Hs, ΣHT)
        S .+= Rs

        # Kalman gain
        CUDA.CUBLAS.getrf_strided_batched!(S, d_ipiv)
        CUDA.CUBLAS.getri_strided_batched!(S, S_inv, d_ipiv)
        NNlib.batched_mul!(K, ΣHT, S_inv)

        # Update
        CUDA.CUBLAS.gemv_strided_batched!('N', 1.0f0, K, y_res, 1.0f0, Aμs)
        # NNlib.batched_vec!(μs, K, y_res; α=1.0f0, β=1.0f0)
        NNlib.batched_mul!(HΣs, Hs, Σs)
        NNlib.batched_mul!(Σs, K, HΣs, -1.0f0, 1.0f0)
    end

    return μs, Σs
end

# Parameters
N = 1000
T = 1000
D = 3

μ0s = CUDA.rand(Float32, D, N)
Σ0s = CUDA.rand(Float32, D, D, N)
Σ0s = NNlib.batched_mul(Σ0s, NNlib.batched_transpose(Σ0s))

As = CUDA.rand(Float32, D, D, N)
bs = CUDA.rand(Float32, D, N)
Qs = CUDA.rand(Float32, D, D, N)
Qs = NNlib.batched_mul(Qs, NNlib.batched_transpose(Qs))

Hs = CUDA.rand(Float32, D, D, N)
cs = CUDA.rand(Float32, D, N)
Rs = CUDA.rand(Float32, D, D, N)
Rs = NNlib.batched_mul(Rs, NNlib.batched_transpose(Rs))

ys = [rand(Float32, D) for _ in 1:T]

μ1, Σ1 = filter_allocations(μ0s, Σ0s, As, bs, Qs, Hs, cs, Rs, ys)
μ2, Σ2 = filter_inplace(μ0s, Σ0s, As, bs, Qs, Hs, cs, Rs, ys)

@benchmark CUDA.@sync filter_allocations($μ0s, $Σ0s, $As, $bs, $Qs, $Hs, $cs, $Rs, $ys)
@benchmark CUDA.@sync filter_inplace($μ0s, $Σ0s, $As, $bs, $Qs, $Hs, $cs, $Rs, $ys)
