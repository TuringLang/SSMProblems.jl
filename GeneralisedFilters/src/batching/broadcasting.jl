using IRTools
using IRTools: @code_ir, IR, Statement, Variable, func
using StructArrays
using LinearAlgebra: I, UniformScaling

using Base.Broadcast: Broadcasted, BroadcastStyle, DefaultArrayStyle
import Base.Broadcast: broadcasted

import PDMats: PDMat

export BATCHED_CACHE_VERBOSITY, clear_batched_cache!

# =============================================================================
# Broadcast Style
# =============================================================================

struct BatchedStyle <: Broadcast.BroadcastStyle end

Base.BroadcastStyle(::Type{<:BatchedCuMatrix}) = BatchedStyle()
Base.BroadcastStyle(::Type{<:BatchedCuVector}) = BatchedStyle()
Base.BroadcastStyle(::Type{<:SharedCuMatrix}) = BatchedStyle()
Base.BroadcastStyle(::Type{<:SharedCuVector}) = BatchedStyle()
Base.BroadcastStyle(::Type{<:StructArray}) = BatchedStyle()
Base.BroadcastStyle(::BatchedStyle, ::BatchedStyle) = BatchedStyle()
Base.BroadcastStyle(::BatchedStyle, ::DefaultArrayStyle{0}) = BatchedStyle()

# =============================================================================
# Ref Conversion (for Shared arrays)
# =============================================================================

maybe_convert_ref(x) = x
maybe_convert_ref(r::Base.RefValue{<:CuVector}) = SharedCuVector(r[])
maybe_convert_ref(r::Base.RefValue{<:CuMatrix}) = SharedCuMatrix(r[])

# =============================================================================
# Structural Operations (Pass-through)
# =============================================================================

broadcasted(::typeof(getproperty), x, s::Symbol) = getproperty(x, s)
broadcasted(::typeof(getfield), x, s::Symbol) = getfield(x, s)
broadcasted(::typeof(getfield), x, i::Int) = getfield(x, i)

# Special handling for RefValue - unwrap before indexing
broadcasted(::typeof(getfield), r::Base.RefValue, i::Int) = getfield(r[], i)
broadcasted(::typeof(getfield), r::Base.RefValue, s::Symbol) = getfield(r[], s)

# StructArray{<:Tuple} destructuring: extract the i-th component array
function broadcasted(::typeof(Base.indexed_iterate), sa::StructArray{<:Tuple}, i::Int)
    return (StructArrays.component(sa, i), i + 1)
end
function broadcasted(
    ::typeof(Base.indexed_iterate), sa::StructArray{<:Tuple}, i::Int, ::Any
)
    return (StructArrays.component(sa, i), i + 1)
end

# =============================================================================
# StructArray Wrapping
# =============================================================================

inner_eltype(arg::BatchedOrShared) = eltype(arg)
inner_eltype(arg::StructArray) = eltype(arg)
inner_eltype(arg) = typeof(arg)

function wrap_if_batched(::Type{T}, args...) where {T}
    if any(arg -> arg isa Union{BatchedOrShared,StructArray}, args)
        field_names = fieldnames(T)
        element_types = Tuple{map(inner_eltype, args)...}
        ElType = Core.Compiler.return_type(T, element_types)
        nt = NamedTuple{field_names}(args)
        return StructArray{ElType}(nt)
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

Returns a StructArray where each element is the type applied to the
corresponding elements of the input arrays.
"""
function broadcasted(::Type{W}, args::Vararg{BatchedOrShared}) where {W}
    element_types = Tuple{map(eltype, args)...}
    ElType = Core.Compiler.return_type(W, element_types)
    field_names = fieldnames(ElType)
    nt = NamedTuple{field_names}(args)
    return StructArray{ElType}(nt)
end

# Redirect function forms to type constructors
broadcasted(::typeof(adjoint), A::BatchedOrShared) = broadcasted(Adjoint, A)
broadcasted(::typeof(transpose), A::BatchedOrShared) = broadcasted(Transpose, A)

# Batched tuple creation: returns StructArray{Tuple{...}}
function broadcasted(::typeof(tuple), args::Vararg{BatchedOrShared})
    ElType = Tuple{map(eltype, args)...}
    return StructArray{ElType}(args)
end

# =============================================================================
# IR Transformation
# =============================================================================

const SKIP_BROADCAST = Set{Any}([getfield, getproperty])

const BROADCAST_TYPES = Set{Any}([PDMat])

# Don't wrap: batched arrays (already batched), callables (functions/types), modules, symbols, integers, already-wrapped refs
maybe_wrap_scalar(x::Union{BatchedOrShared,StructArray}) = x
maybe_wrap_scalar(x::Union{Type,Module,Symbol,Integer,Base.RefValue}) = x
maybe_wrap_scalar(x) = typeof(x) <: Function ? x : Ref(x)

@inline function broadcast_and_materialize(f, args...)
    wrapped_args = map(maybe_wrap_scalar, args)

    # Check if any argument is actually batched (not just Ref-wrapped scalar)
    has_batched = any(arg -> arg isa Union{BatchedOrShared,StructArray}, wrapped_args)

    if !has_batched
        # All scalars - unwrap, execute scalar operation, re-wrap result
        unwrapped_args = map(a -> a isa Base.RefValue ? a[] : a, wrapped_args)
        result = f(unwrapped_args...)
        # Don't wrap code/metadata, batched results, or already-wrapped values
        should_wrap = !(
            typeof(result) <: Function ||
            result isa Union{Type,Module,Symbol,BatchedOrShared,StructArray,Base.RefValue}
        )
        return should_wrap ? Ref(result) : result
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
ir_element_type(::Type{<:StructArray{T}}) where {T} = T
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

function Broadcast.materialize(bc::Broadcasted{BatchedStyle})
    f = bc.f
    args = map(maybe_convert_ref, bc.args)

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
