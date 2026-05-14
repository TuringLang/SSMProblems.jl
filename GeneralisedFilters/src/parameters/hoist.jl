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
    _step_tagged(p::AbstractModelParameter, θ, t, resolved, hoisted_value)

Evaluate a parameter for timestep `t`, **tagging** inactive parameters by wrapping them
back in [`Fixed`](@ref). The tagging propagates dependence information into the type
system so that downstream rrules can dispatch on it and skip gradient computation at
compile time.

- `Fixed`: returns the original `Fixed` object (carries `NoTangent`).
- `FixedParametric`: returns the hoisted plain value (active w.r.t. θ).
- `TimeVarying`: evaluates `p.f(t, resolved)` and wraps in `Fixed`.
- `TimeVaryingParametric`: evaluates `p.f(θ, t, resolved)` as a plain value (active).
"""
_step_tagged(p::Fixed, _, _, _, _) = p
_step_tagged(::FixedParametric, _, _, _, h) = h
_step_tagged(p::TimeVarying, _, t, resolved, _) = Fixed(p.f(t, resolved))
_step_tagged(p::TimeVaryingParametric, θ, t, resolved, _) = p.f(θ, t, resolved)

"""
    _step_eval(p::AbstractModelParameter, θ, t, resolved, hoisted_value)

Evaluate a parameter for timestep `t` and unwrap to a plain value. Used by non-gradient
code paths that do not need the inactive-tagging machinery.
"""
_step_eval(p, θ, t, resolved, h) = _val(_step_tagged(p, θ, t, resolved, h))

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
    return map(_val, step_params(component, nothing, t, resolved, hoist))
end

function step_eval(prior::StatePrior; kwargs...)
    hoist = hoist_static(prior, nothing, (;))
    resolved = NamedTuple(kwargs)
    return map(_val, step_params(prior, nothing, resolved, hoist))
end
