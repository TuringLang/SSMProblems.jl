# The process abstracts, the `distribution`/`simulate`/`logdensity` generics and the
# `StateSpaceModel` container come from SSMProblems. They are re-exported so that
# `using GeneralisedFilters` remains self-sufficient, and so that loading both packages
# does not produce ambiguous bindings.
using SSMProblems: SSMProblems, StatePrior, LatentDynamics, ObservationProcess
using SSMProblems: simulate_from_dist, AbstractStateSpaceModel, prior, dyn, obs
using SSMProblems: DistributionPrior, DistributionDynamics, DistributionObservation
# `import` rather than `using`: these are extended by this package — the generics
# throughout, and `StateSpaceModel` by the hierarchical shorthand constructor.
import SSMProblems: StateSpaceModel, distribution, simulate, logdensity

export StatePrior, LatentDynamics, ObservationProcess
export AbstractStateSpaceModel, StateSpaceModel, prior, dyn, obs
export distribution, simulate, logdensity, simulate_from_dist
export TimeVaryingDynamics, TimeVaryingObservation
export DistributionPrior, DistributionDynamics, DistributionObservation

## CONDITIONING ############################################################################

"""
    resolve(component, ctx)

Resolve a model component against a conditioning context. Constant components (anything in
the process hierarchy) resolve to themselves; any other object is treated as a conditioning
callable and applied to `ctx`, a `NamedTuple` whose fields follow the schema fixed for the
component's slot.

A conditioning callable must not subtype the process abstract types; if it does, it is
treated as a constant.
"""
resolve(prior::StatePrior, ctx) = prior
resolve(dyn::LatentDynamics, ctx) = dyn
resolve(obs::ObservationProcess, ctx) = obs
resolve(f, ctx) = f(ctx)

## TIME-VARYING (NON-RB) WRAPPERS ##########################################################

"""
    TimeVaryingDynamics(f)

Wrap a conditioning callable `f((; t)) -> dynamics` so that it occupies a dynamics slot as a
dispatchable member of the process hierarchy.
"""
struct TimeVaryingDynamics{F} <: LatentDynamics
    f::F
end

"""
    TimeVaryingObservation(f)

Wrap a conditioning callable `f((; t)) -> observation` so that it occupies an observation
slot as a dispatchable member of the process hierarchy.
"""
struct TimeVaryingObservation{F} <: ObservationProcess
    f::F
end

const TimeVarying = Union{TimeVaryingDynamics,TimeVaryingObservation}

function _resolved_component(value, ::Type{P}, slot) where {P}
    _component(value) isa P || throw(
        ArgumentError(
            "$slot must return a $P component, got $(typeof(value)). Use DistributionPrior for a prior distribution, DistributionDynamics/DistributionObservation for distribution-returning functions, or an explicit analytical component for an analytical filter.",
        ),
    )
    return value
end
function resolve(w::TimeVaryingDynamics, ctx)
    return _resolved_component(w.f(ctx), LatentDynamics, "TimeVaryingDynamics")
end
function resolve(w::TimeVaryingObservation, ctx)
    return _resolved_component(w.f(ctx), ObservationProcess, "TimeVaryingObservation")
end
distribution(w::TimeVarying, t::Integer, x) = distribution(resolve(w, (; t)), t, x)
function simulate(rng::AbstractRNG, w::TimeVarying, t::Integer, x)
    return simulate(rng, resolve(w, (; t)), t, x)
end
logdensity(w::TimeVarying, t::Integer, a, b) = logdensity(resolve(w, (; t)), t, a, b)
