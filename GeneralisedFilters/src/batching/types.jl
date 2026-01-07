using CUDA
using LinearAlgebra:
    Adjoint, Transpose, LowerTriangular, UpperTriangular, UniformScaling, Cholesky
using PDMats: PDMat
using StructArrays: StructArray

export BatchedCuMatrix, BatchedCuVector
export SharedCuMatrix, SharedCuVector

# =============================================================================
# Core Batched Types
# =============================================================================

struct BatchedCuMatrix{T,M} <: AbstractVector{CuArray{T,2,M}}
    data::CuArray{T,3,M}
end

struct BatchedCuVector{T,M} <: AbstractVector{CuArray{T,1,M}}
    data::CuArray{T,2,M}
end

const BatchedArray = Union{BatchedCuVector,BatchedCuMatrix}

batch_size(x::BatchedCuVector) = size(x.data, 2)
batch_size(x::BatchedCuMatrix) = size(x.data, 3)

Base.size(x::BatchedCuVector) = (batch_size(x),)
Base.size(x::BatchedCuMatrix) = (batch_size(x),)
Base.length(x::BatchedArray) = batch_size(x)

inner_size(x::BatchedCuVector) = (size(x.data, 1),)
inner_size(x::BatchedCuMatrix) = (size(x.data, 1), size(x.data, 2))

Base.getindex(x::BatchedCuVector, i::Int) = view(x.data, :, i)
Base.getindex(x::BatchedCuMatrix, i::Int) = view(x.data,:,:,i)

# =============================================================================
# Shared Types (same data reused across all batch elements)
# =============================================================================

struct SharedCuMatrix{T,M} <: AbstractVector{CuArray{T,2,M}}
    data::CuArray{T,2,M}
end

struct SharedCuVector{T,M} <: AbstractVector{CuArray{T,1,M}}
    data::CuArray{T,1,M}
end

const SharedArray = Union{SharedCuVector,SharedCuMatrix}


Shared(x::CuArray{T,2,M}) where {T,M} = SharedCuMatrix{T,M}(x)
Shared(x::CuArray{T,1,M}) where {T,M} = SharedCuVector{T,M}(x)

batch_size(::SharedCuVector) = nothing
batch_size(::SharedCuMatrix) = nothing

inner_size(x::SharedCuVector) = size(x.data)
inner_size(x::SharedCuMatrix) = size(x.data)

Base.size(x::SharedCuVector) = (1,)
Base.size(x::SharedCuMatrix) = (1,)
Base.length(::SharedArray) = 1

Base.getindex(x::SharedCuVector, ::Int) = x.data
Base.getindex(x::SharedCuMatrix, ::Int) = x.data

# =============================================================================
# Union Types for Dispatch
# =============================================================================

const BatchedOrShared = Union{
    BatchedCuMatrix,BatchedCuVector,SharedCuMatrix,SharedCuVector,StructArray
}

# =============================================================================
# Helper Functions
# =============================================================================

is_shared(::BatchedCuMatrix) = false
is_shared(::BatchedCuVector) = false
is_shared(::SharedCuMatrix) = true
is_shared(::SharedCuVector) = true

unwrap_data(A::BatchedCuMatrix) = A.data
unwrap_data(A::SharedCuMatrix) = A.data
unwrap_data(x::BatchedCuVector) = x.data
unwrap_data(x::SharedCuVector) = x.data

function inner_size_for_blas(A::BatchedCuMatrix)
    m, n = size(A.data, 1), size(A.data, 2)
    return (m, n)
end

function inner_size_for_blas(A::SharedCuMatrix)
    m, n = size(A.data)
    return (m, n)
end

function get_batch_size(args...)
    for arg in args
        bs = batch_size(arg)
        if bs !== nothing
            return bs
        end
    end
    return error("At least one argument must be batched")
end

# =============================================================================
# Pointer Array Creation
# =============================================================================

function create_pointer_array(A::BatchedCuMatrix{T}) where {T}
    return CUDA.CUBLAS.unsafe_strided_batch(A.data)
end

function create_pointer_array(A::SharedCuMatrix{T}, N::Int) where {T}
    base_ptr = pointer(A.data)
    ptrs_cpu = fill(base_ptr, N)
    return CuArray(ptrs_cpu)
end

function create_pointer_array_vector(x::BatchedCuVector{T}) where {T}
    n = size(x.data, 1)
    N = size(x.data, 2)
    base_ptr = pointer(x.data)
    stride = n * sizeof(T)
    ptrs = CuArray([base_ptr + (i - 1) * stride for i in 1:N])
    return ptrs
end

function create_pointer_array_vector(x::SharedCuVector{T}, N::Int) where {T}
    base_ptr = pointer(x.data)
    ptrs_cpu = fill(base_ptr, N)
    return CuArray(ptrs_cpu)
end
