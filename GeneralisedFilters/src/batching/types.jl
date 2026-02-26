using CUDA
using LinearAlgebra:
    Adjoint, Transpose, LowerTriangular, UpperTriangular, UniformScaling, Cholesky
using PDMats: PDMat

export BatchedCuArray, BatchedCuMatrix, BatchedCuVector
export SharedCuArray, SharedCuMatrix, SharedCuVector, SharedScalar
export Shared
export BatchedStruct

# =============================================================================
# Core Batched Type
# =============================================================================

"""
    BatchedCuArray{T, NE, NB, NT, M} <: AbstractArray{CuArray{T,NE,M}, NB}

An `NB`-dimensional batch of `NE`-dimensional CuArrays, stored as a single contiguous
`CuArray{T, NT, M}` where `NT = NE + NB`.

- `NE`: number of element dimensions (the "inner" array shape)
- `NB`: number of batch dimensions
- `NT`: total number of dimensions (`NE + NB`); required explicitly because Julia's type
  system cannot express arithmetic on type parameters

The first `NE` dimensions index within each element; the last `NB` dimensions index across
the batch.

# Common aliases
- `BatchedCuMatrix{T,M}` = `BatchedCuArray{T,2,1,3,M}` — a vector of matrices
- `BatchedCuVector{T,M}` = `BatchedCuArray{T,1,1,2,M}` — a vector of vectors
"""
struct BatchedCuArray{T,NE,NB,NT,M} <: AbstractArray{CuArray{T,NE,M},NB}
    data::CuArray{T,NT,M}

    function BatchedCuArray{T,NE,NB,NT,M}(data::CuArray{T,NT,M}) where {T,NE,NB,NT,M}
        NE + NB == NT || error("NE ($NE) + NB ($NB) must equal ndims(data) ($NT)")
        return new{T,NE,NB,NT,M}(data)
    end
end

# Convenience constructor: infer T and M, require explicit NE and NB
function BatchedCuArray{T,NE,NB}(data::CuArray{T,NT,M}) where {T,NE,NB,NT,M}
    NE + NB == NT || error("NE ($NE) + NB ($NB) must equal ndims(data) ($NT)")
    return BatchedCuArray{T,NE,NB,NT,M}(data)
end

# Common case aliases
const BatchedCuMatrix{T,M} = BatchedCuArray{T,2,1,3,M}
const BatchedCuVector{T,M} = BatchedCuArray{T,1,1,2,M}

# Constructors for aliased cases
BatchedCuMatrix(data::CuArray{T,3,M}) where {T,M} = BatchedCuArray{T,2,1,3,M}(data)
BatchedCuVector(data::CuArray{T,2,M}) where {T,M} = BatchedCuArray{T,1,1,2,M}(data)

const BatchedArray = BatchedCuArray

Base.IndexStyle(::Type{<:BatchedCuArray}) = Base.IndexCartesian()

function Base.size(x::BatchedCuArray{T,NE,NB}) where {T,NE,NB}
    return ntuple(i -> size(x.data, NE + i), NB)
end

function Base.getindex(x::BatchedCuArray{T,NE,NB}, I::Vararg{Int,NB}) where {T,NE,NB}
    return view(x.data, ntuple(_ -> :, NE)..., I...)
end

function inner_size(x::BatchedCuArray{T,NE}) where {T,NE}
    return ntuple(i -> size(x.data, i), NE)
end

batch_size(x::BatchedCuArray) = length(x)

# =============================================================================
# Shared Types (same data reused across all batch elements)
# =============================================================================

"""
    SharedCuArray{T, InnerN, BatchN, M} <: AbstractArray{CuArray{T,InnerN,M}, BatchN}

A batch of CuArrays where every element is the same underlying `CuArray{T,InnerN,M}`.
Unlike `Ref(array)`, this type carries an explicit batch size and satisfies the
`AbstractArray` contract honestly.

Use `Ref(array)` when the batch size is unknown or irrelevant (e.g. during broadcast
setup). Use `SharedCuArray` when you need a proper `AbstractArray` with a known size.

# Common aliases
- `SharedCuMatrix{T,M}` = `SharedCuArray{T,2,1,M}`
- `SharedCuVector{T,M}` = `SharedCuArray{T,1,1,M}`
"""
struct SharedCuArray{T,InnerN,BatchN,M} <: AbstractArray{CuArray{T,InnerN,M},BatchN}
    data::CuArray{T,InnerN,M}
    batchsize::NTuple{BatchN,Int}
end

# Outer constructor: accept a plain Int for the common 1D-batch case
function SharedCuArray{T,InnerN,1,M}(data::CuArray{T,InnerN,M}, N::Int) where {T,InnerN,M}
    return SharedCuArray{T,InnerN,1,M}(data, (N,))
end

const SharedCuMatrix{T,M} = SharedCuArray{T,2,1,M}
const SharedCuVector{T,M} = SharedCuArray{T,1,1,M}

const SharedArray = SharedCuArray

"""
    Shared(data::CuArray, N::Int) -> SharedCuArray

Convenience constructor: create a `SharedCuArray` from a CuArray with an explicit
1D batch size `N`.
"""
Shared(x::CuArray{T,2,M}, N::Int) where {T,M} = SharedCuArray{T,2,1,M}(x, (N,))
Shared(x::CuArray{T,1,M}, N::Int) where {T,M} = SharedCuArray{T,1,1,M}(x, (N,))

Base.IndexStyle(::Type{<:SharedCuArray}) = Base.IndexCartesian()

Base.size(x::SharedCuArray) = x.batchsize

function Base.getindex(
    x::SharedCuArray{T,InnerN,BatchN}, ::Vararg{Int,BatchN}
) where {T,InnerN,BatchN}
    return x.data
end

function inner_size(x::SharedCuArray)
    return size(x.data)
end

batch_size(x::SharedCuArray) = length(x)

# =============================================================================
# SharedScalar: a scalar value shared across all batch elements
# =============================================================================

struct SharedScalar{T} <: AbstractVector{T}
    value::T
end

Base.size(::SharedScalar) = (1,)
Base.length(::SharedScalar) = 1
Base.getindex(x::SharedScalar, ::Int) = x.value
Base.eltype(::Type{SharedScalar{T}}) where {T} = T
batch_size(::SharedScalar) = nothing
_get_component_batch_size(::SharedScalar) = nothing

# Comparisons unwrap the SharedScalar automatically
Base.:(==)(x::SharedScalar, y) = x.value == y
Base.:(==)(x, y::SharedScalar) = x == y.value
Base.:(==)(x::SharedScalar, y::SharedScalar) = x.value == y.value

# =============================================================================
# BatchedStruct - Custom wrapper for batched composite types
# =============================================================================

"""
    BatchedStruct{T, C <: NamedTuple} <: AbstractVector{T}

A wrapper type representing a batch of structs of type `T`, stored in a
column-oriented (struct-of-arrays) format.

# Type Parameters
- `T`: The element type (e.g., `PDMat{Float32, CuMatrix{Float32}}`)
- `C`: The NamedTuple type holding the batched components

# Fields
- `components::C`: A NamedTuple where each field is a batched array or nested BatchedStruct

# Usage
BatchedStruct is designed to be created automatically by the batching IR transform
when constructors are called with batched arguments. Users typically don't need
to construct these directly.

# Property Access
- `x.fieldname` returns the batched component for real fields of `T`
- `x.components` returns the underlying NamedTuple storage
- For computed properties (custom getproperty), falls back to element-wise evaluation

# Indexing
- `x[i]` constructs and returns an instance of `T` for the i-th batch element
"""
struct BatchedStruct{T,C<:NamedTuple} <: AbstractVector{T}
    components::C

    function BatchedStruct{T}(components::C) where {T,C<:NamedTuple}
        # Validate all components have consistent batch size
        sizes = map(_get_component_batch_size, values(components))
        non_nothing = Base.filter(!isnothing, sizes)
        if !isempty(non_nothing)
            first_size = first(non_nothing)
            if !all(s -> s == first_size, non_nothing)
                error("All batched components must have the same batch size")
            end
        end
        return new{T,C}(components)
    end
end

# Helper to get batch size from a component
_get_component_batch_size(x::BatchedCuArray) = batch_size(x)
_get_component_batch_size(x::SharedCuArray) = batch_size(x)
_get_component_batch_size(x::BatchedStruct) = length(x)
_get_component_batch_size(::Any) = nothing

# Convenience constructor that infers T from the fieldnames matching a type
function BatchedStruct{T}(; kwargs...) where {T}
    components = NamedTuple{fieldnames(T)}(values(kwargs))
    return BatchedStruct{T}(components)
end

# =============================================================================
# BatchedStruct - AbstractVector Interface
# =============================================================================

function batch_size(x::BatchedStruct)
    for component in values(getfield(x, :components))
        bs = _get_component_batch_size(component)
        if bs !== nothing
            return bs
        end
    end
    return error("BatchedStruct has no batched components")
end

Base.size(x::BatchedStruct) = (batch_size(x),)
Base.IndexStyle(::Type{<:BatchedStruct}) = IndexLinear()

# Indexing: construct an element of type T by calling its constructor
@generated function Base.getindex(x::BatchedStruct{T}, i::Integer) where {T}
    fields = fieldnames(T)
    field_exprs = [:(getfield(x, :components).$(fields[j])[i]) for j in 1:length(fields)]
    return :(T($(field_exprs...)))
end

# =============================================================================
# BatchedStruct - Property Access
# =============================================================================

function Base.getproperty(x::BatchedStruct{T}, s::Symbol) where {T}
    s === :components && return getfield(x, :components)
    s in fieldnames(T) && return getfield(x, :components)[s]
    # Computed property - broadcast getproperty, triggering IR transformation
    return getproperty.(x, Ref(s))
end

Base.propertynames(::BatchedStruct{T}) where {T} = (:components, fieldnames(T)...)

# =============================================================================
# BatchedStruct - Display (avoid materialization)
# =============================================================================

function Base.show(io::IO, x::BatchedStruct{T}) where {T}
    return print(io, "BatchedStruct{", T, "} with ", length(x), " elements")
end

function Base.show(io::IO, ::MIME"text/plain", x::BatchedStruct{T}) where {T}
    println(io, "BatchedStruct{", T, "} with ", length(x), " elements:")
    comps = getfield(x, :components)
    for name in keys(comps)
        component = comps[name]
        print(io, "  .", name, " :: ")
        if component isa BatchedCuArray
            println(io, typeof(component), " (", inner_size(component), ")")
        elseif component isa SharedCuArray
            println(io, typeof(component), " [shared] (", inner_size(component), ")")
        elseif component isa BatchedStruct
            println(io, typeof(component))
        else
            println(io, typeof(component))
        end
    end
end

# =============================================================================
# Union Types for Dispatch
# =============================================================================

const BatchedOrShared = Union{BatchedCuArray,SharedCuArray,BatchedStruct}

# =============================================================================
# Helper Functions
# =============================================================================

is_shared(::BatchedCuArray) = false
is_shared(::SharedCuArray) = true

unwrap_data(A::BatchedCuArray) = A.data
unwrap_data(A::SharedCuArray) = A.data

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

function create_pointer_array(A::BatchedCuArray{T,InnerN,1}) where {T,InnerN}
    return CUDA.CUBLAS.unsafe_strided_batch(A.data)
end

function create_pointer_array(A::SharedCuArray{T,InnerN,1}) where {T,InnerN}
    N = batch_size(A)
    ptr = pointer(A.data)
    return reinterpret(CuPtr{T}, CUDA.fill(UInt(ptr), N))
end
