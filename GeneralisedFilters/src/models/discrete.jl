export DiscretePrior, DiscreteLatentDynamics
export DiscreteStateSpaceModel

export calc_α0, calc_P

import SSMProblems: distribution
import Distributions: Categorical

## MODEL COMPONENTS ########################################################################

"""
    DiscretePrior(α0)

Categorical prior over the initial state. `α0` is an [`AbstractModelParameter`](@ref)
holding the probability vector; raw vectors are auto-wrapped via [`as_parameter`](@ref).
"""
struct DiscretePrior{AP<:AbstractModelParameter} <: StatePrior
    α0::AP
end
DiscretePrior(α0) = DiscretePrior(as_parameter(α0))

"""
    DiscreteLatentDynamics(P)

Discrete latent dynamics with transition matrix `P[i, j] = p(x_t = j | x_{t-1} = i)`.
`P` is an [`AbstractModelParameter`](@ref); raw matrices are auto-wrapped via
[`as_parameter`](@ref).
"""
struct DiscreteLatentDynamics{PP<:AbstractModelParameter} <: LatentDynamics
    P::PP
end
DiscreteLatentDynamics(P) = DiscreteLatentDynamics(as_parameter(P))

const DiscreteStateSpaceModel = SSMProblems.StateSpaceModel{
    <:DiscretePrior,<:DiscreteLatentDynamics,<:ObservationProcess
}

## HOIST / STEP-PARAM DISPATCH #############################################################

function hoist_static(p::DiscretePrior, θ, hoisted_controls)
    return (α0=_hoist(p.α0, θ, hoisted_controls),)
end

function step_params(p::DiscretePrior, θ, hoisted_controls, hoist)
    return (α0=_step_tagged(p.α0, θ, 0, hoisted_controls, hoist.α0),)
end

function hoist_static(c::DiscreteLatentDynamics, θ, hoisted_controls)
    return (P=_hoist(c.P, θ, hoisted_controls),)
end

function step_params(c::DiscreteLatentDynamics, θ, t, resolved, hoist)
    return (P=_step_tagged(c.P, θ, t, resolved, hoist.P),)
end

## CALC-* SHIMS ############################################################################

calc_α0(p::DiscretePrior; kwargs...) = _eval_param(p.α0, 0, kwargs)
calc_P(dyn::DiscreteLatentDynamics, t::Integer; kwargs...) = _eval_param(dyn.P, t, kwargs)

## SSMPROBLEMS DISTRIBUTIONS ###############################################################

function SSMProblems.distribution(prior::DiscretePrior; kwargs...)
    return Categorical(calc_α0(prior; kwargs...))
end

function SSMProblems.distribution(
    dyn::DiscreteLatentDynamics, step::Integer, state::Integer; kwargs...
)
    P = calc_P(dyn, step; kwargs...)
    return Categorical(P[state, :])
end
