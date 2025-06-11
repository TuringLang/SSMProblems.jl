import Base: *, +, -, transpose, getindex
import LinearAlgebra: Transpose, cholesky, \, /, I, UniformScaling, dot
import Distributions: logpdf
import Random: rand

export BatchedCuVector, BatchedCuMatrix, BatchedCuCholesky

###########################
#### VECTOR OPERATIONS ####
###########################

struct BatchedCuVector{T} <: BatchedVector{T}
    data::CuArray{T,2,CUDA.DeviceMemory}
    ptrs::CuVector{CuPtr{T},CUDA.DeviceMemory}
end
function BatchedCuVector(data::CuArray{T,2}) where {T}
    ptrs = CUDA.CUBLAS.unsafe_strided_batch(data)
    return BatchedCuVector{T}(data, ptrs)
end
Base.eltype(::BatchedCuVector{T}) where {T} = T
Base.length(x::BatchedCuVector) = size(x.data, 2)

function +(x::BatchedCuVector{T}, y::BatchedCuVector{T}) where {T}
    z_data = x.data .+ y.data
    return BatchedCuVector(z_data)
end

function -(x::BatchedCuVector{T}, y::BatchedCuVector{T}) where {T}
    z_data = x.data .- y.data
    return BatchedCuVector(z_data)
end

function dot(x::BatchedCuVector{T}, y::BatchedCuVector{T}) where {T}
    if size(x.data, 1) != size(y.data, 1)
        throw(DimensionMismatch("Vectors must have the same length for dot product"))
    end
    xy = x.data .* y.data
    return dropdims(sum(xy; dims=1); dims=1)
end

###########################
#### MATRIX OPERATIONS ####
###########################

struct BatchedCuMatrix{T} <: BatchedMatrix{T}
    data::CuArray{T,3,CUDA.DeviceMemory}
    ptrs::CuVector{CuPtr{T},CUDA.DeviceMemory}
end
function BatchedCuMatrix(data::CuArray{T,3}) where {T}
    ptrs = CUDA.CUBLAS.unsafe_strided_batch(data)
    return BatchedCuMatrix{T}(data, ptrs)
end
Base.eltype(::BatchedCuMatrix{T}) where {T} = T
Base.length(A::BatchedCuMatrix) = size(A.data, 3)

transpose(A::BatchedCuMatrix{T}) where {T} = Transpose{T,BatchedCuMatrix{T}}(A)

function *(A::BatchedCuMatrix{T}, B::BatchedCuMatrix{T}) where {T}
    C_data = CUDA.CUBLAS.gemm_strided_batched('N', 'N', A.data, B.data)
    return BatchedCuMatrix(C_data)
end
function *(A::Transpose{T,BatchedCuMatrix{T}}, B::BatchedCuMatrix{T}) where {T}
    C_data = CUDA.CUBLAS.gemm_strided_batched('T', 'N', A.parent.data, B.data)
    return BatchedCuMatrix(C_data)
end
function *(A::BatchedCuMatrix{T}, B::Transpose{T,BatchedCuMatrix{T}}) where {T}
    C_data = CUDA.CUBLAS.gemm_strided_batched('N', 'T', A.data, B.parent.data)
    return BatchedCuMatrix(C_data)
end
function *(A::Transpose{T,BatchedCuMatrix{T}}, B::Transpose{T,BatchedCuMatrix{T}}) where {T}
    C_data = CUDA.CUBLAS.gemm_strided_batched('T', 'T', A.parent.data, B.parent.data)
    return BatchedCuMatrix(C_data)
end

function +(A::BatchedCuMatrix{T}, B::BatchedCuMatrix{T}) where {T}
    C_data = A.data .+ B.data
    return BatchedCuMatrix(C_data)
end
function +(A::BatchedCuMatrix{T}, B::Transpose{T,BatchedCuMatrix{T}}) where {T}
    C_data = A.data .+ permutedims(B.parent.data, (2, 1, 3))
    return BatchedCuMatrix(C_data)
end

function -(A::BatchedCuMatrix{T}, B::BatchedCuMatrix{T}) where {T}
    C_data = A.data .- B.data
    return BatchedCuMatrix(C_data)
end

function +(A::BatchedCuMatrix{T}, J::UniformScaling{<:Union{T,Bool}}) where {T}
    m, n = size(A.data, 1), size(A.data, 2)
    m == n || throw(DimensionMismatch("Matrix must be square for UniformScaling addition"))
    B_data = copy(A.data)
    for i in 1:m
        B_data[i, i, :] .+= J.λ
    end
    return BatchedCuMatrix(B_data)
end

##################################
#### MATRIX-VECTOR OPERATIONS ####
##################################

function *(A::BatchedCuMatrix{T}, x::BatchedCuVector{T}) where {T}
    y_data = CuArray{T}(undef, size(A.data, 1), size(x.data, 2))
    CUDA.CUBLAS.gemv_strided_batched!('N', T(1.0), A.data, x.data, T(0.0), y_data)
    return BatchedCuVector(y_data)
end
function *(A::Transpose{T,BatchedCuMatrix{T}}, x::BatchedCuVector{T}) where {T}
    y_data = CuArray{T}(undef, size(A.parent.data, 2), size(x.data, 2))
    CUDA.CUBLAS.gemv_strided_batched!('T', T(1.0), A.parent.data, x.data, T(0.0), y_data)
    return BatchedCuVector(y_data)
end

function getindex(A::BatchedCuMatrix{T}, i::Int, ::Colon) where {T}
    row_data = A.data[i, :, :]
    return BatchedCuVector(row_data)
end
function getindex(A::BatchedCuMatrix{T}, ::Colon, j::Int) where {T}
    col_data = A.data[:, j, :]
    return BatchedCuVector(col_data)
end

###########################
#### SCALAR OPERATIONS ####
###########################

function /(A::BatchedCuMatrix, s::Number)
    C_data = A.data ./ s
    return BatchedCuMatrix(C_data)
end

#########################
#### POTR OPERATIONS ####
#########################

struct BatchedCuCholesky{T} <: BatchedMatrix{T}
    data::CuArray{T,3,CUDA.DeviceMemory}
    ptrs::CuVector{CuPtr{T},CUDA.DeviceMemory}
end
function BatchedCuCholesky(data::CuArray{T,3}) where {T}
    ptrs = CUDA.CUBLAS.unsafe_strided_batch(data)
    return BatchedCuCholesky{T}(data, ptrs)
end
Base.eltype(::BatchedCuCholesky{T}) where {T} = T
Base.length(P::BatchedCuCholesky) = size(P.data, 3)

for (fname, elty) in (
    (:cusolverDnSpotrfBatched, :Float32),
    (:cusolverDnDpotrfBatched, :Float64),
    (:cusolverDnCpotrfBatched, :ComplexF32),
    (:cusolverDnZpotrfBatched, :ComplexF64),
)
    @eval begin
        function cholesky(A::BatchedCuMatrix{$elty})
            # HACK: assuming A is positive definite
            m, n, b = size(A.data)
            m == n ||
                throw(DimensionMismatch("Matrix must be square for Cholesky decomposition"))

            P_data = copy(A.data)
            P = BatchedCuCholesky(P_data)

            dh = CUDA.CUSOLVER.dense_handle()
            info = CuVector{Int}(undef, b)
            CUDA.CUSOLVER.$fname(dh, 'L', m, P.ptrs, m, info, b)

            return P
        end
    end
end

# TODO: CUSOLVER does not support matrix RHS for potrs; replace with MAGMA
for (fname, elty) in (
    (:cusolverDnSpotrsBatched, :Float32),
    (:cusolverDnDpotrsBatched, :Float64),
    (:cusolverDnCpotrsBatched, :ComplexF32),
    (:cusolverDnZpotrsBatched, :ComplexF64),
)
    @eval begin
        function \(P::BatchedCuCholesky{$elty}, A::BatchedCuMatrix{$elty})
            m, n, b, = size(A.data)
            # CUSOLVER does not support matrix RHS for potrs so solve each column separately
            bs_data = Vector{CuMatrix{$elty}}(undef, n)
            for i in 1:n
                bs_data[i] = A.data[:, i, :]
                b_ptr = CUDA.CUBLAS.unsafe_strided_batch(bs_data[i])

                dh = CUDA.CUSOLVER.dense_handle()
                info = CuVector{Int}(undef, b)
                CUDA.CUSOLVER.$fname(dh, 'L', m, 1, P.ptrs, m, b_ptr, m, info, b)
            end

            B_data = stack(bs_data; dims=2)
            return BatchedCuMatrix(B_data)
        end
    end
end

for (fname, elty) in (
    (:cusolverDnSpotrsBatched, :Float32),
    (:cusolverDnDpotrsBatched, :Float64),
    (:cusolverDnCpotrsBatched, :ComplexF32),
    (:cusolverDnZpotrsBatched, :ComplexF64),
)
    @eval begin
        function \(P::BatchedCuCholesky{$elty}, x::BatchedCuVector{$elty})
            m, b = size(x.data)
            y_data = copy(x.data)
            y = BatchedCuVector(y_data)
            dh = CUDA.CUSOLVER.dense_handle()
            info = CuVector{Int}(undef, b)
            CUDA.CUSOLVER.$fname(dh, 'L', m, 1, P.ptrs, m, y.ptrs, m, info, b)
            return y
        end
    end
end

##########################
#### MIXED OPERATIONS ####
##########################

function -(x::CuVector{T}, y::BatchedCuVector{T}) where {T}
    z_data = x .- y.data
    return BatchedCuVector(z_data)
end
function +(x::CuVector{T}, y::BatchedCuVector{T}) where {T}
    z_data = x .+ y.data
    return BatchedCuVector(z_data)
end
function +(x::BatchedCuVector{T}, y::CuVector{T}) where {T}
    z_data = x.data .+ y
    return BatchedCuVector(z_data)
end
# TODO: these need to be generated automatically and call a common function
# TODO: are we best using the strided or non-strided version. The former don't need pointer duplication
for (fname, elty, gemv_batched) in (
    (:cublasSgemvBatched_64, :Float32, CUDA.CUBLAS.cublasSgemvBatched_64),
    (:cublasDgemvBatched_64, :Float64, CUDA.CUBLAS.cublasDgemvBatched_64),
    (:cublasCgemvBatched_64, :ComplexF32, CUDA.CUBLAS.cublasCgemvBatched_64),
    (:cublasZgemvBatched_64, :ComplexF64, CUDA.CUBLAS.cublasZgemvBatched_64),
)
    @eval begin
        function *(A::BatchedCuMatrix{$elty}, x::CuVector{$elty})
            m, n, b = size(A.data)
            y_data = CuArray{$elty}(undef, m, b)
            y = BatchedCuVector(y_data)

            # Call gemv directly
            x_ptrs = batch_singleton(x, b)
            h = CUDA.CUBLAS.handle()
            $gemv_batched(
                h, 'N', m, n, $elty(1.0), A.ptrs, m, x_ptrs, 1, $elty(0.0), y.ptrs, 1, b
            )
            return y
        end

        function *(A::CuMatrix{$elty}, x::BatchedCuVector{$elty})
            m, n = size(A)
            b = size(x.data, 2)
            y_data = CuArray{$elty}(undef, m, b)
            y = BatchedCuVector(y_data)

            # Call gemv directly
            A_ptrs = batch_singleton(A, b)
            h = CUDA.CUBLAS.handle()
            $gemv_batched(
                h, 'N', m, n, $elty(1.0), A_ptrs, m, x.ptrs, 1, $elty(0.0), y.ptrs, 1, b
            )
            return y
        end
    end
end

@inline function batch_singleton(array::DenseCuArray{T}, N::Int) where {T}
    ptrs = CuArray{CuPtr{T}}(undef, N)
    function compute_pointers()
        i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
        grid_stride = gridDim().x * blockDim().x
        while i <= length(ptrs)
            @inbounds ptrs[i] = reinterpret(CuPtr{T}, pointer(array))
            i += grid_stride
        end
        return nothing
    end
    kernel = @cuda launch = false compute_pointers()
    config = launch_configuration(kernel.fun)
    threads = min(config.threads, N)
    blocks = min(config.blocks, cld(N, threads))
    @cuda threads blocks compute_pointers()
    return ptrs
end

###################################
#### DISTRIBUTIONAL OPERATIONS ####
###################################

# Can likely replace by using Gaussian
function gaussian_likelihood(
    m::BatchedCuVector{T}, S::BatchedCuMatrix{T}, y::Union{BatchedCuVector{T},CuVector{T}}
) where {T}
    D = size(S.data, 1)
    y_res = y - m

    # TODO: avoid recomputing Cholesky decomposition
    S_chol = cholesky(S)

    diags = CuArray{T}(undef, size(S.data, 1), size(S.data, 3))
    for i in 1:size(S.data, 1)
        diags[i, :] = S_chol.data[i, i, :]
    end
    log_dets = T(2) * dropdims(sum(log.(diags); dims=1); dims=1)

    inv_term = S_chol \ y_res
    log_likes = -T(0.5) * dot(y_res, inv_term)
    log_likes .-= T(0.5) * (log_dets .+ D * log(T(2π)))

    # HACK: only errors seems to be from numerical stability so will just overwrite
    log_likes[isnan.(log_likes)] .= -Inf

    return log_likes
end

# HACK: this is more hard-coded than it needs to be. Can generalise to other batched types
# by taking advantage of the internal calls used in GaussianDistributions.jl
function Distributions.logpdf(
    P::Gaussian{BatchedCuVector{T},BatchedCuMatrix{T}},
    y::Union{BatchedCuVector{T},CuVector{T}},
) where {T}
    return gaussian_likelihood(P.μ, P.Σ, y)
end
# HACK: MAJOR — this is just to handle a special case for bootstrap filter unit test until
# we have a general approach to this
function Distributions.logpdf(
    P::Gaussian{BatchedCuVector{T},<:CuMatrix{T}}, y::CuVector{T}
) where {T}
    # Stack Σ to form a batched matrix
    Σ_data = CuArray{T}(undef, size(P.Σ)..., size(P.μ.data, 2))
    Σ_data[:, :, :] .= P.Σ
    return gaussian_likelihood(P.μ, BatchedCuMatrix(Σ_data), y)
end

# TODO: need to generalise to only one argument being batched
function Random.rand(
    ::AbstractRNG, P::Gaussian{BatchedCuVector{T},BatchedCuMatrix{T}}
) where {T}
    D, N = size(P.μ.data)
    Σ_chol = cholesky(P.Σ)
    Z = BatchedCuVector(CUDA.randn(T, D, N))
    # HACK: CUBLAS doesn't have batched trmv so we'll use gemm with zeroing out for now.
    # Should later replace with MAGMA
    L = BatchedCuMatrix(Σ_chol.data)
    zero_upper_triangle!(L.data)
    return P.μ + L * Z
end
# TODO: the singleton Cholesky should probably be handled on the CPU
function Random.rand(::AbstractRNG, P::Gaussian{BatchedCuVector{T},<:CuMatrix{T}}) where {T}
    D, N = size(P.μ.data)
    Σ_L = cholesky(P.Σ).L
    Z = BatchedCuVector(CUDA.randn(T, D, N))
    return P.μ + CuArray(Σ_L) * Z
end

function zero_upper_triangle!(A::CuArray{T,3}) where {T}
    D, _, N = size(A)

    function kernel_zero_upper_triangle!(A, D)
        i = threadIdx().x
        j = threadIdx().y
        k = blockIdx().x

        if i < j && i <= D && j <= D
            A[i, j, k] = zero(eltype(A))
        end
        return nothing
    end

    @cuda threads = (D, D) blocks = N kernel_zero_upper_triangle!(A, D)
    return nothing
end
