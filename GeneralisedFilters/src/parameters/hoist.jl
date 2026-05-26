"""
    _hoist(p::AbstractModelParameter, θ, hoisted_controls)

Pre-evaluate the part of a parameter `p` that does not depend on `t`. Returns the raw
value for [`Fixed`](@ref) and [`FixedParametric`](@ref); returns `nothing` for time-varying
traits. `hoisted_controls` is the resolved `NamedTuple` produced by [`hoist_controls`](@ref).
"""
_hoist(p::Fixed, _, _) = p.value
_hoist(p::FixedParametric, θ, hoisted) = p.f(θ, hoisted)
_hoist(::TimeVarying, _, _) = nothing
_hoist(::TimeVaryingParametric, _, _) = nothing

"""
    _step_eval(p::AbstractModelParameter, θ, t, resolved, hoisted_value)

Evaluate a parameter for timestep `t` and return the plain (unwrapped) value.

- `Fixed`: returns the wrapped value.
- `FixedParametric`: returns the pre-evaluated hoisted value.
- `TimeVarying`: evaluates `p.f(t, resolved)`.
- `TimeVaryingParametric`: evaluates `p.f(θ, t, resolved)`.

Trait-driven gradient routing happens in the `ssm_loglikelihood` rrule by dispatching
on the model component's field types directly (not on the evaluated value's type), so
no value-level "Fixed" tagging is needed here.
"""
_step_eval(p::Fixed, _, _, _, _) = p.value
_step_eval(::FixedParametric, _, _, _, h) = h
_step_eval(p::TimeVarying, _, t, resolved, _) = p.f(t, resolved)
_step_eval(p::TimeVaryingParametric, θ, t, resolved, _) = p.f(θ, t, resolved)

"""
    step_eval(component, t::Integer; kwargs...)
    step_eval(prior::StatePrior; kwargs...)

Evaluate the parameters of a model component at timestep `t` and return a `NamedTuple` of
plain (unwrapped) values. Keyword arguments are treated as the resolved-controls
`NamedTuple`. Used by non-gradient algorithm paths; gradient paths go through
[`step_params`](@ref) directly.

Only valid for components whose parameters do not depend on θ (Fixed/TimeVarying).
"""
function step_eval(component, t::Integer; kwargs...)
    hoist = hoist_static(component, nothing, (;))
    resolved = NamedTuple(kwargs)
    return step_params(component, nothing, t, resolved, hoist)
end

function step_eval(prior::StatePrior; kwargs...)
    hoist = hoist_static(prior, nothing, (;))
    resolved = NamedTuple(kwargs)
    return step_params(prior, nothing, resolved, hoist)
end
