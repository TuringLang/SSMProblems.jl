using CUDA
using LinearAlgebra:
    Adjoint, Transpose, LowerTriangular, UpperTriangular, UniformScaling, Cholesky
using PDMats: PDMat

export BatchedCuMatrix, BatchedCuVector
export SharedCuMatrix, SharedCuVector
export BatchedPDMat, BatchedCholesky

# =============================================================================
# Core Batched Types
# =============================================================================

struct BatchedCuMatrix{T,Inner<:AbstractMatrix{T}} <: AbstractVector{Inner}
    data::CuArray{T,3}
end

struct BatchedCuVector{T,Inner<:AbstractVector{T}} <: AbstractVector{Inner}
    data::CuMatrix{T}
end

const BatchedArray = Union{BatchedCuVector,BatchedCuMatrix}

BatchedCuMatrix(data::CuArray{T,3}) where {T} = BatchedCuMatrix{T,CuMatrix{T}}(data)
BatchedCuVector(data::CuMatrix{T}) where {T} = BatchedCuVector{T,CuVector{T}}(data)

batch_size(x::BatchedCuVector) = size(x.data, 2)
batch_size(x::BatchedCuMatrix) = size(x.data, 3)

Base.size(x::BatchedCuVector) = (batch_size(x),)
Base.size(x::BatchedCuMatrix) = (batch_size(x),)
Base.length(x::BatchedArray) = batch_size(x)

inner_size(x::BatchedCuVector) = (size(x.data, 1),)
inner_size(x::BatchedCuMatrix) = (size(x.data, 1), size(x.data, 2))

function Base.getindex(x::BatchedCuVector{T,CuVector{T}}, i::Int) where {T}
    return view(x.data, :, i)
end

function Base.getindex(x::BatchedCuMatrix{T,CuMatrix{T}}, i::Int) where {T}
    return view(x.data,:,:,i)
end

function Base.getindex(
    x::BatchedCuMatrix{T,LowerTriangular{T,CuMatrix{T}}}, i::Int
) where {T}
    return LowerTriangular(view(x.data,:,:,i))
end

function Base.getindex(
    x::BatchedCuMatrix{T,UpperTriangular{T,CuMatrix{T}}}, i::Int
) where {T}
    return UpperTriangular(view(x.data,:,:,i))
end

function Base.getindex(x::BatchedCuMatrix{T,Adjoint{T,CuMatrix{T}}}, i::Int) where {T}
    return adjoint(view(x.data,:,:,i))
end

function Base.getindex(x::BatchedCuMatrix{T,Transpose{T,CuMatrix{T}}}, i::Int) where {T}
    return transpose(view(x.data,:,:,i))
end

# =============================================================================
# Shared Types (same data reused across all batch elements)
# =============================================================================

struct SharedCuMatrix{T,Inner<:AbstractMatrix{T}} <: AbstractVector{Inner}
    data::CuMatrix{T}
end

struct SharedCuVector{T,Inner<:AbstractVector{T}} <: AbstractVector{Inner}
    data::CuVector{T}
end

const SharedArray = Union{SharedCuVector,SharedCuMatrix}

SharedCuMatrix(data::CuMatrix{T}) where {T} = SharedCuMatrix{T,CuMatrix{T}}(data)
SharedCuVector(data::CuVector{T}) where {T} = SharedCuVector{T,CuVector{T}}(data)

Shared(x::CuMatrix{T}) where {T} = SharedCuMatrix(x)
Shared(x::CuVector{T}) where {T} = SharedCuVector(x)

batch_size(::SharedCuVector) = nothing
batch_size(::SharedCuMatrix) = nothing

inner_size(x::SharedCuVector) = size(x.data)
inner_size(x::SharedCuMatrix) = size(x.data)

Base.size(x::SharedCuVector) = (1,)
Base.size(x::SharedCuMatrix) = (1,)
Base.length(::SharedArray) = 1

Base.getindex(x::SharedCuVector, ::Int) = x.data
Base.getindex(x::SharedCuMatrix{T,CuMatrix{T}}, ::Int) where {T} = x.data
function Base.getindex(x::SharedCuMatrix{T,LowerTriangular{T,CuMatrix{T}}}, ::Int) where {T}
    return LowerTriangular(x.data)
end

# =============================================================================
# Type Aliases and Union Types for Dispatch
# =============================================================================

const AnyBatchedMatrix{T} = Union{
    BatchedCuMatrix{T,CuMatrix{T}},
    BatchedCuMatrix{T,Adjoint{T,CuMatrix{T}}},
    BatchedCuMatrix{T,Transpose{T,CuMatrix{T}}},
    BatchedCuMatrix{T,LowerTriangular{T,CuMatrix{T}}},
    BatchedCuMatrix{T,UpperTriangular{T,CuMatrix{T}}},
}

const AnySharedMatrix{T} = Union{
    SharedCuMatrix{T,CuMatrix{T}},
    SharedCuMatrix{T,Adjoint{T,CuMatrix{T}}},
    SharedCuMatrix{T,Transpose{T,CuMatrix{T}}},
    SharedCuMatrix{T,LowerTriangular{T,CuMatrix{T}}},
    SharedCuMatrix{T,UpperTriangular{T,CuMatrix{T}}},
}

const AnyMatrix{T} = Union{AnyBatchedMatrix{T},AnySharedMatrix{T}}
const AnyVector{T} = Union{BatchedCuVector{T},SharedCuVector{T}}

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

trans_flag(::BatchedCuMatrix{T,CuMatrix{T}}) where {T} = 'N'
trans_flag(::BatchedCuMatrix{T,Adjoint{T,CuMatrix{T}}}) where {T} = T <: Real ? 'T' : 'C'
trans_flag(::BatchedCuMatrix{T,Transpose{T,CuMatrix{T}}}) where {T} = 'T'
trans_flag(::BatchedCuMatrix{T,LowerTriangular{T,CuMatrix{T}}}) where {T} = 'N'
trans_flag(::BatchedCuMatrix{T,UpperTriangular{T,CuMatrix{T}}}) where {T} = 'N'

trans_flag(::SharedCuMatrix{T,CuMatrix{T}}) where {T} = 'N'
trans_flag(::SharedCuMatrix{T,Adjoint{T,CuMatrix{T}}}) where {T} = T <: Real ? 'T' : 'C'
trans_flag(::SharedCuMatrix{T,Transpose{T,CuMatrix{T}}}) where {T} = 'T'
trans_flag(::SharedCuMatrix{T,LowerTriangular{T,CuMatrix{T}}}) where {T} = 'N'
trans_flag(::SharedCuMatrix{T,UpperTriangular{T,CuMatrix{T}}}) where {T} = 'N'

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
# Stateful Wrapper Types (Cholesky, PDMat)
# =============================================================================

struct BatchedCholesky{T} <: AbstractVector{Cholesky{T,CuMatrix{T}}}
    factors::BatchedCuMatrix{T,LowerTriangular{T,CuMatrix{T}}}
    info::CuVector{Int32}
    uplo::Char
end

struct BatchedPDMat{T} <: AbstractVector{PDMat{T,CuMatrix{T}}}
    chol::BatchedCholesky{T}
end

batch_size(c::BatchedCholesky) = batch_size(c.factors)
batch_size(p::BatchedPDMat) = batch_size(p.chol)

inner_size(c::BatchedCholesky) = inner_size(c.factors)
inner_size(p::BatchedPDMat) = inner_size(p.chol)

Base.size(c::BatchedCholesky) = (batch_size(c),)
Base.size(p::BatchedPDMat) = (batch_size(p),)

Base.getindex(p::BatchedPDMat{T}, i::Int) where {T} = p.chol.factors[i] * p.chol.factors[i]'

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
