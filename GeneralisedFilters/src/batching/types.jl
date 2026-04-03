using Adapt
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
    BatchedCuArray{T,NE,NB,NT,A<:AbstractArray{T,NT}} <: AbstractArray{Any,NB}

An `NB`-dimensional batch of `NE`-dimensional arrays, stored as a single contiguous
`NT`-dimensional array `data` where `NT = NE + NB`.

- `NE`: number of element dimensions (the "inner" array shape)
- `NB`: number of batch dimensions
- `NT`: total number of dimensions (`NE + NB`); required explicitly because Julia's type
  system cannot express arithmetic on type parameters
- `A`: storage array type

The first `NE` dimensions index within each element; the last `NB` dimensions index across
the batch.

This type is generic over the storage array type so that it can participate in `Adapt.jl`
transformations. In the user-facing intended usage, `data` is a `CuArray{T, NT, M}`.

# Common aliases
- `BatchedCuMatrix{T,A}` = `BatchedCuArray{T,2,1,3,A}` — a vector of matrices
- `BatchedCuVector{T,A}` = `BatchedCuArray{T,1,1,2,A}` — a vector of vectors
"""
struct BatchedCuArray{T,NE,NB,NT,A<:AbstractArray{T,NT}} <: AbstractArray{Any,NB}
    data::A

    function BatchedCuArray{T,NE,NB,NT,A}(data::A) where {T,NE,NB,NT,A<:AbstractArray{T,NT}}
        NE + NB == NT || error("NE ($NE) + NB ($NB) must equal ndims(data) ($NT)")
        return new{T,NE,NB,NT,A}(data)
    end
end

# Convenience constructor: infer T and M, require explicit NE and NB
function BatchedCuArray{T,NE,NB}(data::A) where {T,NE,NB,A<:AbstractArray{T}}
    NT = ndims(data)
    NE + NB == NT || error("NE ($NE) + NB ($NB) must equal ndims(data) ($NT)")
    return BatchedCuArray{T,NE,NB,NT,A}(data)
end

# Common case aliases
const BatchedCuMatrix{T,A<:AbstractArray{T,3}} = BatchedCuArray{T,2,1,3,A}
const BatchedCuVector{T,A<:AbstractArray{T,2}} = BatchedCuArray{T,1,1,2,A}

# Constructors for aliased cases
BatchedCuMatrix(data::A) where {T,A<:AbstractArray{T,3}} = BatchedCuArray{T,2,1,3,A}(data)
BatchedCuVector(data::A) where {T,A<:AbstractArray{T,2}} = BatchedCuArray{T,1,1,2,A}(data)

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

# Adapting BatchedCuArray to bitstype
function Adapt.adapt_structure(
    to,
    x::BatchedCuArray{T,NE,NB,NT,A},
) where {T,NE,NB,NT,A}
    data_adapted = Adapt.adapt(to, x.data)
    return BatchedCuArray{T,NE,NB,NT,typeof(data_adapted)}(data_adapted)
end

# =============================================================================
# Shared Types (same data reused across all batch elements)
# =============================================================================

"""
    SharedCuArray{T,InnerN,BatchN,A<:AbstractArray{T,InnerN}} <: AbstractArray{Any,BatchN}

A batch of arrays where every element is the same underlying array.
Unlike `Ref(array)`, this type carries an explicit batch size and satisfies the
`AbstractArray` contract honestly.

Use `Ref(array)` when the batch size is unknown or irrelevant (e.g. during broadcast
setup). Use `SharedCuArray` when you need a proper `AbstractArray` with a known size.

This type is generic over the storage array type so that it can participate in `Adapt.jl`
transformations. In the user-facing intended usage, `data` is a `CuArray{T,InnerN,M}`.

# Common aliases
- `SharedCuMatrix{T,A}` = `SharedCuArray{T,2,1,A}`
- `SharedCuVector{T,A}` = `SharedCuArray{T,1,1,A}`
"""
struct SharedCuArray{T,InnerN,BatchN,A<:AbstractArray{T,InnerN}} <: AbstractArray{Any,BatchN}
    data::A
    batchsize::NTuple{BatchN,Int}
end

# Outer constructor: accept a plain Int for the common 1D-batch case
function SharedCuArray{T,InnerN,1,A}(data::A, N::Int) where {T,InnerN,A<:AbstractArray{T,InnerN}}
    return SharedCuArray{T,InnerN,1,A}(data, (N,))
end

const SharedCuMatrix{T,A<:AbstractArray{T,2}} = SharedCuArray{T,2,1,A}
const SharedCuVector{T,A<:AbstractArray{T,1}} = SharedCuArray{T,1,1,A}

# Constructors for aliased cases
SharedCuMatrix(data::A, N::Int) where {T,A<:AbstractArray{T,2}} = SharedCuArray{T,2,1,A}(data, N)
SharedCuVector(data::A, N::Int) where {T,A<:AbstractArray{T,1}} = SharedCuArray{T,1,1,A}(data, N)

const SharedArray = SharedCuArray

Base.eltype(::Type{<:BatchedCuArray{T,NE}}) where {T,NE} = AbstractArray{T,NE}
Base.eltype(::Type{<:SharedCuArray{T,InnerN}}) where {T,InnerN} = AbstractArray{T,InnerN}

"""
    Shared(data::AbstractArray, N::Int) -> SharedCuArray

Convenience constructor: create a `SharedCuArray` from an arrat with an explicit
1D batch size `N`.

The underlying storage is generic to support `Adapt.jl` transformations, but in
the user-facing intended interface `A` is type `CuArray`
"""
Shared(x::A, N::Int) where {T,A<:AbstractArray{T,2}} = SharedCuArray{T,2,1,A}(x, (N,))
Shared(x::A, N::Int) where {T,A<:AbstractArray{T,1}} = SharedCuArray{T,1,1,A}(x, (N,))

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

# Adapting SharedCuArray to bitstype
function Adapt.adapt_structure(
    to,
    x::SharedCuArray{T,InnerN,BatchN,A},
) where {T,InnerN,BatchN,A<:AbstractArray{T,InnerN}}
    data_adapted = Adapt.adapt(to, x.data)
    return SharedCuArray{T,InnerN,BatchN,typeof(data_adapted)}(data_adapted, x.batchsize)
end

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

# Adapting SharedScalar to bitstype
Adapt.@adapt_structure SharedScalar

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

# Non-generic indexing
function Base.getindex(x::BatchedStruct{<:Adjoint}, i::Integer)
    return adjoint(getfield(x, :components).parent[i])
end

function Base.getindex(x::BatchedStruct{<:Transpose}, i::Integer)
    return transpose(getfield(x, :components).parent[i])
end

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

# Adapting BatchedStruct to bitstype
function Adapt.adapt_structure(
    to,
    x::BatchedStruct{T,C},
) where {T,C<:NamedTuple}
    comps_adapted = Adapt.adapt(to, x.components)
    return BatchedStruct{T}(comps_adapted)
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
