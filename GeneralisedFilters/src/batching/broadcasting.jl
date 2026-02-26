using IRTools
using IRTools: @code_ir, IR, Statement, Variable, func
using LinearAlgebra: I, UniformScaling

using Base.Broadcast: Broadcasted, BroadcastStyle, DefaultArrayStyle
import Base.Broadcast: broadcasted

import PDMats: PDMat

export BATCHED_CACHE_VERBOSITY, clear_batched_cache!

# =============================================================================
# Broadcast Style
# =============================================================================

struct BatchedStyle <: Broadcast.BroadcastStyle end

Base.BroadcastStyle(::Type{<:BatchedCuArray}) = BatchedStyle()
Base.BroadcastStyle(::Type{<:SharedCuArray}) = BatchedStyle()
Base.BroadcastStyle(::Type{<:BatchedStruct}) = BatchedStyle()
Base.BroadcastStyle(::BatchedStyle, ::BatchedStyle) = BatchedStyle()
Base.BroadcastStyle(::BatchedStyle, ::DefaultArrayStyle{0}) = BatchedStyle()

# =============================================================================
# Ref Conversion (for Shared arrays)
# =============================================================================

maybe_convert_ref(x, ::Nothing) = x
maybe_convert_ref(x, ::Int) = x
function maybe_convert_ref(r::Base.RefValue{<:CuVector{T,M}}, N::Int) where {T,M}
    return SharedCuVector{T,M}(r[], (N,))
end
function maybe_convert_ref(r::Base.RefValue{<:CuMatrix{T,M}}, N::Int) where {T,M}
    return SharedCuMatrix{T,M}(r[], (N,))
end
# Can't convert without knowing N — leave as Ref and let downstream handle it
maybe_convert_ref(r::Base.RefValue{<:CuVector}, ::Nothing) = r
maybe_convert_ref(r::Base.RefValue{<:CuMatrix}, ::Nothing) = r

# BatchedStruct{<:Tuple} destructuring: extract the i-th component
function broadcasted(::typeof(Base.indexed_iterate), bs::BatchedStruct{<:Tuple}, i::Int)
    return (getfield(bs, :components)[i], i + 1)
end
function broadcasted(
    ::typeof(Base.indexed_iterate), bs::BatchedStruct{<:Tuple}, i::Int, ::Any
)
    return (getfield(bs, :components)[i], i + 1)
end

# =============================================================================
# BatchedStruct Wrapping
# =============================================================================

inner_eltype(arg::BatchedOrShared) = eltype(arg)
inner_eltype(arg) = typeof(arg)

"""
    wrap_if_batched(::Type{T}, args...)

If any argument is batched, create a BatchedStruct{T} with the args as components.
Otherwise, call the constructor T(args...) normally.
"""
function wrap_if_batched(::Type{T}, args...) where {T}
    if any(arg -> arg isa BatchedOrShared, args)
        field_names = fieldnames(T)
        element_types = Tuple{map(inner_eltype, args)...}
        ElType = Core.Compiler.return_type(T, element_types)
        nt = NamedTuple{field_names}(args)
        return BatchedStruct{ElType}(nt)
    else
        return T(args...)
    end
end

# =============================================================================
# Generic Wrapper Broadcasting
# =============================================================================

"""
    broadcasted(::Type{W}, args::BatchedOrShared...)

Generic wrapper for any type constructor applied to batched arguments.
Works for single-field wrappers (Adjoint, Transpose, LowerTriangular, etc.)
as well as multi-field types (PDMat, Cholesky, etc.).

Returns a BatchedStruct where each element is the type applied to the
corresponding elements of the input arrays.
"""
function broadcasted(::Type{W}, args::Vararg{BatchedOrShared}) where {W}
    element_types = Tuple{map(eltype, args)...}
    ElType = Core.Compiler.return_type(W, element_types)
    field_names = fieldnames(ElType)
    nt = NamedTuple{field_names}(args)
    return BatchedStruct{ElType}(nt)
end

# Redirect function forms to type constructors
broadcasted(::typeof(adjoint), A::BatchedOrShared) = broadcasted(Adjoint, A)
broadcasted(::typeof(transpose), A::BatchedOrShared) = broadcasted(Transpose, A)

# copy for Adjoint/Transpose wrappers - materialize the transposition
function broadcasted(::typeof(copy), x::BatchedStruct{<:Adjoint})
    parent_data = x.parent  # BatchedCuMatrix or SharedCuMatrix
    if parent_data isa BatchedCuMatrix
        return BatchedCuMatrix(permutedims(parent_data.data, (2, 1, 3)))
    else  # SharedCuMatrix
        return SharedCuMatrix(permutedims(parent_data.data, (2, 1)))
    end
end

function broadcasted(::typeof(copy), x::BatchedStruct{<:Transpose})
    parent_data = x.parent
    if parent_data isa BatchedCuMatrix
        return BatchedCuMatrix(permutedims(parent_data.data, (2, 1, 3)))
    else  # SharedCuMatrix
        return SharedCuMatrix(permutedims(parent_data.data, (2, 1)))
    end
end

# Union of all types that represent batched data
const BatchedData = Union{BatchedOrShared,CuVector}

# Batched tuple creation: returns BatchedStruct{Tuple{...}}
function broadcasted(::typeof(tuple), args::Vararg{BatchedData})
    ElType = Tuple{map(eltype, args)...}
    # For tuples, components is a regular Tuple, not NamedTuple
    components = NamedTuple{ntuple(i -> Symbol("x$i"), length(args))}(args)
    return BatchedStruct{ElType}(components)
end

# =============================================================================
# getfield/getproperty broadcasting for BatchedStruct
# =============================================================================

# getfield on BatchedStruct: return the batched component
# Handle both unwrapped and Ref-wrapped field names (Ref from maybe_wrap_scalar)
function broadcasted(::typeof(getfield), x::BatchedStruct{T}, s::Symbol) where {T}
    s in fieldnames(T) && return getfield(x, :components)[s]
    return error("BatchedStruct{$T} has no field `$s`")
end
function broadcasted(
    ::typeof(getfield), x::BatchedStruct{T}, s::Base.RefValue{Symbol}
) where {T}
    return broadcasted(getfield, x, s[])
end
broadcasted(::typeof(getfield), x::BatchedStruct, i::Int) = getfield(x, :components)[i]
function broadcasted(::typeof(getfield), x::BatchedStruct, i::Base.RefValue{<:Integer})
    return getfield(x, :components)[i[]]
end

# getproperty on BatchedStruct: return batched component for real fields,
# fall through to tracing for computed properties
function broadcasted(::typeof(getproperty), x::BatchedStruct{T}, s::Symbol) where {T}
    s in fieldnames(T) && return getfield(x, :components)[s]
    # Computed property - return Broadcasted to trigger IR transformation
    return Broadcasted{BatchedStyle}(getproperty, (x, s))
end

function broadcasted(
    ::typeof(getproperty), x::BatchedStruct{T}, s::Base.RefValue{Symbol}
) where {T}
    return broadcasted(getproperty, x, s[])
end

# =============================================================================
# size broadcasting for batched arrays (returns inner dimensions)
# =============================================================================

broadcasted(::typeof(size), A::BatchedCuMatrix) = inner_size(A)
broadcasted(::typeof(size), A::BatchedCuMatrix, i::Integer) = inner_size(A)[i]
broadcasted(::typeof(size), A::SharedCuMatrix) = inner_size(A)
broadcasted(::typeof(size), A::SharedCuMatrix, i::Integer) = inner_size(A)[i]

broadcasted(::typeof(size), x::BatchedCuVector) = inner_size(x)
broadcasted(::typeof(size), x::BatchedCuVector, i::Integer) = inner_size(x)[i]
broadcasted(::typeof(size), x::SharedCuVector) = inner_size(x)
broadcasted(::typeof(size), x::SharedCuVector, i::Integer) = inner_size(x)[i]

# =============================================================================
# IR Transformation
# =============================================================================

const SKIP_BROADCAST = Set{Any}()
const BROADCAST_TYPES = Set{Any}([PDMat])

# Don't wrap: batched data, shared scalars, callables, modules, symbols, integers, already-wrapped refs
maybe_wrap_scalar(x::BatchedData) = x
maybe_wrap_scalar(x::SharedScalar) = x
maybe_wrap_scalar(x::Union{Type,Module,Symbol,Integer,Base.RefValue}) = x
maybe_wrap_scalar(x) = typeof(x) <: Function ? x : Ref(x)

@inline function broadcast_and_materialize(f, args...)
    wrapped_args = map(maybe_wrap_scalar, args)

    # Check if any argument is actually batched
    has_batched = any(arg -> arg isa BatchedData, wrapped_args)

    if !has_batched
        # All scalars - unwrap and execute directly
        unwrapped_args = map(a -> a isa Base.RefValue ? a[] : a, wrapped_args)
        return f(unwrapped_args...)
    end

    # Special case: getfield on Ref-wrapped scalar object (getfield is a builtin, can't trace)
    if f === getfield && length(wrapped_args) >= 1
        obj = wrapped_args[1]
        if obj isa Base.RefValue
            return getfield(obj[], wrapped_args[2:end]...)
        end
    end

    # Has batched inputs - normal broadcast path
    result = broadcasted(f, wrapped_args...)
    if result isa Broadcasted
        return Broadcast.materialize(result)
    end
    return result
end

function resolve_to_type(ir::IR, val)
    val isa Type && return val
    if val isa Variable
        stmt = ir[val]
        if stmt !== nothing
            return resolve_to_type(ir, stmt.expr)
        end
    end
    if val isa GlobalRef
        try
            resolved = getfield(val.mod, val.name)
            return resolved isa Type ? resolved : nothing
        catch
            return nothing
        end
    end
    return nothing
end

function is_type_ref(ir::IR, val)
    return resolve_to_type(ir, val) !== nothing
end

function is_broadcast_type(ir::IR, val)
    resolved = resolve_to_type(ir, val)
    return resolved !== nothing && resolved in BROADCAST_TYPES
end

function transform_to_batched(ir::IR)
    ir = copy(ir)

    for (v, stmt) in ir
        if stmt.expr isa Expr && stmt.expr.head == :call
            fn = stmt.expr.args[1]
            if fn in SKIP_BROADCAST
                continue
            end
            if is_broadcast_type(ir, fn)
                new_args = [broadcast_and_materialize, stmt.expr.args...]
                ir[v] = Statement(stmt; expr=Expr(:call, new_args...))
                continue
            end
            if is_type_ref(ir, fn)
                new_args = [wrap_if_batched, stmt.expr.args...]
                ir[v] = Statement(stmt; expr=Expr(:call, new_args...))
                continue
            end
            new_args = [broadcast_and_materialize, stmt.expr.args...]
            ir[v] = Statement(stmt; expr=Expr(:call, new_args...))
        end
    end

    return ir
end

ir_element_type(::Type{T}) where {T} = T
ir_element_type(::Type{<:BatchedStruct{T}}) where {T} = T
ir_element_type(::Type{<:Base.RefValue{T}}) where {T} = T

function generate_batched_function(f, argtypes::Type{<:Tuple})
    element_types = Tuple{map(ir_element_type, argtypes.parameters)...}

    ir = IRTools.Inner.code_ir(f, element_types)
    if ir === nothing
        error(
            "Could not get IR for function $f with types $element_types (original: $argtypes)",
        )
    end
    batched_ir = transform_to_batched(ir)
    return IRTools.func(batched_ir)
end

# =============================================================================
# Broadcast Materialization
# =============================================================================

# Verbosity levels: :silent, :verbose, :debug
# :silent  - no output
# :verbose - print when generating or regenerating (i.e. cache misses)
# :debug   - print all cache activity including hits
const BATCHED_CACHE_VERBOSITY = Ref{Symbol}(:silent)

# Cache stores (batched_function, world_age_when_cached)
const BATCHED_FUNC_CACHE = Dict{Tuple{Any,Type},Tuple{Any,UInt}}()

"""
    clear_batched_cache!()

Clear the batched function cache. Useful for debugging or forcing regeneration.
"""
function clear_batched_cache!()
    empty!(BATCHED_FUNC_CACHE)
    return nothing
end

function _find_batch_size(args)
    for arg in args
        if arg isa BatchedCuArray || arg isa SharedCuArray
            return batch_size(arg)
        elseif arg isa Broadcasted
            n = _find_batch_size(arg.args)
            n !== nothing && return n
        end
    end
    return nothing
end

function Broadcast.materialize(bc::Broadcasted{BatchedStyle})
    f = bc.f
    N = _find_batch_size(bc.args)
    args = map(a -> maybe_convert_ref(a, N), bc.args)

    result = broadcasted(f, args...)
    if !(result isa Broadcasted)
        return result
    end

    argtypes = Tuple{map(typeof, args)...}
    key = (f, argtypes)

    # Get element types for method lookup
    element_types = Tuple{map(ir_element_type, argtypes.parameters)...}

    if haskey(BATCHED_FUNC_CACHE, key)
        batched_f, cached_world = BATCHED_FUNC_CACHE[key]
        # Check if the method has been redefined since caching
        m = which(f, element_types)
        if m.primary_world <= cached_world
            if BATCHED_CACHE_VERBOSITY[] == :debug
                println("  [Using cached batched version of $f]")
            end
            return Base.invokelatest(batched_f, nothing, args...)
        end
        if BATCHED_CACHE_VERBOSITY[] in (:verbose, :debug)
            println("  [Regenerating batched version of $f (method redefined)]")
        end
    else
        if BATCHED_CACHE_VERBOSITY[] in (:verbose, :debug)
            println("  [Generating batched version of $f]")
        end
    end

    batched_f = generate_batched_function(f, argtypes)
    current_world = Base.get_world_counter()
    BATCHED_FUNC_CACHE[key] = (batched_f, current_world)
    return Base.invokelatest(batched_f, nothing, args...)
end
