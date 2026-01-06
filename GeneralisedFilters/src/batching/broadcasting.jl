using IRTools
using IRTools: @code_ir, IR, Statement, Variable, func
using StructArrays
using LinearAlgebra: I, UniformScaling

using Base.Broadcast: Broadcasted, BroadcastStyle, DefaultArrayStyle
import Base.Broadcast: broadcasted

import PDMats: PDMat

# =============================================================================
# Broadcast Style
# =============================================================================

struct BatchedStyle <: Broadcast.BroadcastStyle end

Base.BroadcastStyle(::Type{<:BatchedCuMatrix}) = BatchedStyle()
Base.BroadcastStyle(::Type{<:BatchedCuVector}) = BatchedStyle()
Base.BroadcastStyle(::Type{<:SharedCuMatrix}) = BatchedStyle()
Base.BroadcastStyle(::Type{<:SharedCuVector}) = BatchedStyle()
Base.BroadcastStyle(::Type{<:BatchedCholesky}) = BatchedStyle()
Base.BroadcastStyle(::Type{<:BatchedPDMat}) = BatchedStyle()
# HACK: Currently hard-coded but can be replaced with a custom StructArray type
Base.BroadcastStyle(::Type{<:StructArray}) = BatchedStyle()
Base.BroadcastStyle(::BatchedStyle, ::BatchedStyle) = BatchedStyle()
Base.BroadcastStyle(::BatchedStyle, ::DefaultArrayStyle{0}) = BatchedStyle()

# =============================================================================
# Ref Conversion (for Shared arrays)
# =============================================================================

maybe_convert_ref(x) = x
function maybe_convert_ref(r::Base.RefValue{<:CuVector{T}}) where {T}
    return SharedCuVector{T,CuVector{T}}(r[])
end
function maybe_convert_ref(r::Base.RefValue{<:CuMatrix{T}}) where {T}
    return SharedCuMatrix{T,CuMatrix{T}}(r[])
end

# =============================================================================
# Structural Operations (Pass-through)
# =============================================================================

broadcasted(::typeof(tuple), args...) = tuple(args...)
broadcasted(::typeof(getproperty), x, s::Symbol) = getproperty(x, s)
broadcasted(::typeof(getfield), x, s::Symbol) = getfield(x, s)
broadcasted(::typeof(getfield), x, i::Int) = getfield(x, i)

# Special handling for RefValue - unwrap before indexing
broadcasted(::typeof(getfield), r::Base.RefValue, i::Int) = getfield(r[], i)
broadcasted(::typeof(getfield), r::Base.RefValue, s::Symbol) = getfield(r[], s)

# =============================================================================
# StructArray Wrapping
# =============================================================================

inner_eltype(arg::BatchedCuVector{T}) where {T} = CuVector{T}
inner_eltype(arg::BatchedCuMatrix{T}) where {T} = CuMatrix{T}
inner_eltype(arg::SharedCuVector{T}) where {T} = CuVector{T}
inner_eltype(arg::SharedCuMatrix{T}) where {T} = CuMatrix{T}
inner_eltype(arg::BatchedPDMat{T}) where {T} = PDMat{T,CuMatrix{T}}
inner_eltype(arg) = typeof(arg)

function wrap_if_batched(::Type{T}, args...) where {T}
    if any(arg -> arg isa Union{BatchedArray,SharedArray,BatchedPDMat}, args)
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
# IR Transformation
# =============================================================================

const SKIP_BROADCAST = Set{Any}([
    tuple,
    Core.tuple,
    getfield,
    getproperty,
    adjoint,
    transpose,
    LowerTriangular,
    UpperTriangular,
])

const BROADCAST_TYPES = Set{Any}([PDMat])

maybe_wrap_scalar(x) = x
maybe_wrap_scalar(x::UniformScaling) = Ref(x)

@inline function broadcast_and_materialize(f, args...)
    wrapped_args = map(maybe_wrap_scalar, args)
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

const BATCHED_FUNC_CACHE = Dict{Tuple,Any}()

function Broadcast.materialize(bc::Broadcasted{BatchedStyle})
    f = bc.f
    args = map(maybe_convert_ref, bc.args)

    result = broadcasted(f, args...)
    if !(result isa Broadcasted)
        return result
    end

    argtypes = Tuple{map(typeof, args)...}
    key = (f, argtypes)

    if !haskey(BATCHED_FUNC_CACHE, key)
        println("  [Generating batched version of $f]")
        batched_f = generate_batched_function(f, argtypes)
        BATCHED_FUNC_CACHE[key] = batched_f
    end

    batched_f = BATCHED_FUNC_CACHE[key]
    return Base.invokelatest(batched_f, nothing, args...)
end
