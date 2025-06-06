import Base: *, +, -, transpose, getindex
import LinearAlgebra: Transpose, cholesky, \, /, I, UniformScaling, dot

export BatchedCuVector, BatchedCuMatrix, BatchedCuCholesky

abstract type BatchedVector{T} end
abstract type BatchedMatrix{T} end

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

###################################
#### DISTRIBUTIONAL OPERATIONS ####
###################################

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
