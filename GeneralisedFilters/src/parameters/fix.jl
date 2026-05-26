export fix

"""
    fix(model, θ) -> non_parametric_model
    fix(component, θ) -> non_parametric_component
    fix(param, θ) -> Fixed or TimeVarying

Replace every [`FixedParametric`](@ref) / [`TimeVaryingParametric`](@ref) parameter in
a model with its non-parametric counterpart at the given `θ`. The result is a model
whose parameters are all `Fixed` or `TimeVarying` — suitable for any code path that
doesn't carry `θ` (e.g. conditional SMC, sampling, plain forward filtering).

`fix` does **not** preserve gradients with respect to `θ`: a `FixedParametric` becomes
a `Fixed` holding an eagerly evaluated value. For gradient paths use
[`ssm_loglikelihood`](@ref) directly, passing `θ` explicitly.

The fix path assumes the model has no parametric controls. Parametric parameters'
`f(θ, hoisted_controls)` is called with an empty `hoisted_controls = (;)`.
"""
function fix end

## PARAMETERS ##############################################################################

fix(p::Fixed, _) = p
fix(p::TimeVarying, _) = p
fix(p::FixedParametric, θ) = Fixed(p.f(θ, (;)))

function fix(p::TimeVaryingParametric, θ)
    f = p.f
    return TimeVarying((t, c) -> f(θ, t, c))
end

## MODELS ##################################################################################
# Component-level `fix` dispatches are defined alongside the model components themselves
# (linear_gaussian.jl, discrete.jl, hierarchical.jl) since they know each component's
# field layout. The generic `StateSpaceModel` dispatch composes them.

function fix(m::SSMProblems.StateSpaceModel, θ)
    return SSMProblems.StateSpaceModel(
        fix(SSMProblems.prior(m), θ),
        fix(SSMProblems.dyn(m), θ),
        fix(SSMProblems.obs(m), θ),
    )
end
