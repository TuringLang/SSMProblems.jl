export hoist_controls, resolve_controls

"""
    hoist_controls(controls::NamedTuple, θ) -> NamedTuple

Pre-evaluate the entries of a user-supplied controls `NamedTuple` that do not depend on `t`.
[`Fixed`](@ref) and [`FixedParametric`](@ref) entries are evaluated; [`TimeVarying`](@ref)
and [`TimeVaryingParametric`](@ref) entries return `nothing` (they are resolved per step).

For v1, control function signatures do not receive other controls — each control is
evaluated independently.
"""
function hoist_controls(controls::NamedTuple, θ)
    return map(p -> _hoist_control(p, θ), controls)
end

_hoist_control(p::Fixed, _) = p.value
_hoist_control(p::FixedParametric, θ) = p.f(θ)
_hoist_control(::TimeVarying, _) = nothing
_hoist_control(::TimeVaryingParametric, _) = nothing

"""
    resolve_controls(controls::NamedTuple, hoisted::NamedTuple, θ, t) -> NamedTuple

Produce a `NamedTuple` of fully-resolved control values for timestep `t`, reusing entries
already evaluated by [`hoist_controls`](@ref) and evaluating the time-varying entries.
"""
function resolve_controls(controls::NamedTuple, hoisted::NamedTuple, θ, t)
    return map((p, h) -> _step_eval_control(p, θ, t, h), controls, hoisted)
end

_step_eval_control(::Fixed, _, _, h) = h
_step_eval_control(::FixedParametric, _, _, h) = h
_step_eval_control(p::TimeVarying, _, t, _) = p.f(t)
_step_eval_control(p::TimeVaryingParametric, θ, t, _) = p.f(θ, t)
