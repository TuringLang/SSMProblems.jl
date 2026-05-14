export AbstractModelParameter
export Fixed, FixedParametric, TimeVarying, TimeVaryingParametric
export as_parameter

"""
    AbstractModelParameter

Supertype for parameter wrappers that declare how a model component value depends on the
inference parameters `θ` and the time index `t`. The wrapper type is a full dependence
contract — it determines whether gradients are computed for the value:

| Trait                  | θ-dependent? | t-dependent? |
|------------------------|--------------|--------------|
| [`Fixed`](@ref)                   | No           | No           |
| [`FixedParametric`](@ref)         | Yes          | No           |
| [`TimeVarying`](@ref)             | **No**       | Yes          |
| [`TimeVaryingParametric`](@ref)   | Yes          | Yes          |

`TimeVarying` means the parameter does not depend on θ *anywhere in its evaluation chain*
— including indirectly through other controls. If a parameter reads a control whose value
depends on θ, it must be declared `TimeVaryingParametric`. Misdeclaring as `TimeVarying`
silently zeros the gradient contribution along that path.
"""
abstract type AbstractModelParameter end

"""
    Fixed(value)

Parameter that does not depend on θ or t. The wrapped value is used directly.
"""
struct Fixed{T} <: AbstractModelParameter
    value::T
end

"""
    FixedParametric(f)

Parameter that depends on θ but not t. `f(θ, hoisted_controls)` is evaluated once before
the filtering loop (where `hoisted_controls` is the resolved `NamedTuple` of non-time-
varying controls). For controls themselves, the signature is `f(θ)` — see [`hoist_controls`](@ref).
"""
struct FixedParametric{F} <: AbstractModelParameter
    f::F
end

"""
    TimeVarying(f)

Parameter that depends on t but not on θ (anywhere in its evaluation chain). `f(t, resolved)`
is evaluated each timestep, where `resolved` is the per-step resolved controls. For controls,
the signature is `f(t)`.
"""
struct TimeVarying{F} <: AbstractModelParameter
    f::F
end

"""
    TimeVaryingParametric(f)

Parameter that depends on both θ and t. `f(θ, t, resolved)` is evaluated each timestep.
For controls, the signature is `f(θ, t)`.
"""
struct TimeVaryingParametric{F} <: AbstractModelParameter
    f::F
end

"""
    as_parameter(x)

Wrap a raw value as a [`Fixed`](@ref) parameter. Pass-through for values that already
subtype [`AbstractModelParameter`](@ref). Used by convenience constructors that accept
either matrices/vectors or parameter wrappers.
"""
as_parameter(p::AbstractModelParameter) = p
as_parameter(x) = Fixed(x)

"""
    _val(x)

Unwrap a [`Fixed`](@ref) tag to its underlying value. Pass-through for plain (non-
parameter) values. Used inside `_kalman_step` and analogous primitives to obtain the
raw matrix/vector arguments from possibly-tagged parameters.

Calling `_val` on any other [`AbstractModelParameter`](@ref) is a programming error —
parameters that reach the primitive must already have been resolved by `_step_tagged`.
"""
_val(x::Fixed) = x.value
function _val(x::AbstractModelParameter)
    return error(
        "_val called on an unresolved $(typeof(x)); parameters must be resolved via _step_tagged before reaching the primitive",
    )
end
_val(x) = x
