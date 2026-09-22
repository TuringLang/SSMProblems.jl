export HierarchicalPrior, HierarchicalDynamics, HierarchicalObservation, HierarchicalSSM
export HierarchicalState
export inner_prior, inner_dynamics, inner_observation, condition_inner

"""
    HierarchicalPrior(outer, inner)

Prior for a Rao-Blackwellisable model. `outer` is the prior for the sampled component;
`inner` is either a constant prior atom or a conditioning callable `(; x0) -> atom`.
"""
struct HierarchicalPrior{OP<:StatePrior,IP} <: StatePrior
    outer::OP
    inner::IP
end

"""
    HierarchicalDynamics(outer, inner)

Dynamics for a Rao-Blackwellisable model. `outer` is the dynamics for the sampled component;
`inner` is either a constant dynamics atom or a conditioning callable
`(; t, x_prev, x_new) -> atom`.
"""
struct HierarchicalDynamics{OD<:LatentDynamics,ID} <: LatentDynamics
    outer::OD
    inner::ID
end

"""
    HierarchicalObservation(inner)

Observation process for a Rao-Blackwellisable model. `inner` is either a constant observation
atom/emission or a conditioning callable `(; t, x) -> atom/emission`.
"""
struct HierarchicalObservation{IO} <: ObservationProcess
    inner::IO
end

const HierarchicalSSM = StateSpaceModel{
    <:HierarchicalPrior,<:HierarchicalDynamics,<:HierarchicalObservation
}

"""
    StateSpaceModel(outer_prior, outer_dyn, inner_prior, inner_dyn, obs)

Shorthand for building a hierarchical model. The explicit component form using
`HierarchicalPrior`/`HierarchicalDynamics`/`HierarchicalObservation` is the canonical
construction.
"""
function StateSpaceModel(
    outer_prior::StatePrior, outer_dyn::LatentDynamics, inner_prior, inner_dyn, obs
)
    return StateSpaceModel(
        HierarchicalPrior(outer_prior, inner_prior),
        HierarchicalDynamics(outer_dyn, inner_dyn),
        HierarchicalObservation(obs),
    )
end

"""
    HierarchicalState(x, z)

A joint sample from a hierarchical model, with outer component `x` and a sampled inner
component `z`. This differs from an `RBState`, whose inner component is a distribution rather
than a sample.
"""
struct HierarchicalState{XT,ZT}
    x::XT
    z::ZT
end

## CONDITIONAL COMPONENT INTERFACE ########################################################

"""
    inner_prior(model::HierarchicalSSM, x0)
    inner_prior(prior::HierarchicalPrior, x0)
    inner_prior(component, x0)

Resolve the inner prior conditioned on the initial outer state. Define conditioning through
callables or `resolve` methods; the model-level methods are convenience delegates. The
component form uses `resolve(component, (; x0))`; hierarchical components and models
delegate to it.
"""
inner_prior(component, x0) = resolve(component, (; x0))
inner_prior(p::HierarchicalPrior, x0) = inner_prior(p.inner, x0)
inner_prior(m::HierarchicalSSM, x0) = inner_prior(m.prior, x0)

"""
    inner_dynamics(model::HierarchicalSSM, t, x_prev, x_new)
    inner_dynamics(dynamics::HierarchicalDynamics, t, x_prev, x_new)
    inner_dynamics(component, t, x_prev, x_new)

Resolve the inner transition at time `t` using the adjacent outer states. Both generative
hierarchical operations and conditioned likelihoods use this component-level operation.
Longer history dependence requires an augmented outer state for consistent incremental
reuse in particle filtering.
"""
function inner_dynamics(component, t::Integer, x_prev, x_new)
    return resolve(component, (; t, x_prev, x_new))
end
function inner_dynamics(d::HierarchicalDynamics, t::Integer, x_prev, x_new)
    return inner_dynamics(d.inner, t, x_prev, x_new)
end
function inner_dynamics(m::HierarchicalSSM, t::Integer, x_prev, x_new)
    return inner_dynamics(m.dyn, t, x_prev, x_new)
end

"""
    inner_observation(model::HierarchicalSSM, t, x)
    inner_observation(observation::HierarchicalObservation, t, x)
    inner_observation(component, t, x)

Resolve the inner observation process at time `t` conditioned on the current outer state.
The component form uses `resolve(component, (; t, x))`.
"""
inner_observation(component, t::Integer, x) = resolve(component, (; t, x))
function inner_observation(o::HierarchicalObservation, t::Integer, x)
    return inner_observation(o.inner, t, x)
end
inner_observation(m::HierarchicalSSM, t::Integer, x) = inner_observation(m.obs, t, x)

# Only ReferenceTrajectory uses physical time as its array index. Ordinary vectors store
# x0 at index 1. Reject other axes rather than silently interpreting an offset vector.
function _validate_trajectory(xs::AbstractVector)
    isempty(xs) && throw(ArgumentError("a trajectory must include its initial state"))
    Base.require_one_based_indexing(xs)
    return nothing
end
function _validate_trajectory(xs::ReferenceTrajectory)
    Base.require_one_based_indexing(xs.xs)
    return nothing
end
_trajectory_state(xs::AbstractVector, t::Integer) = xs[t + 1]
_trajectory_state(xs::ReferenceTrajectory, t::Integer) = xs[t]

function _validate_trajectory_time(xs, t::Integer)
    1 <= t < length(xs) || throw(
        ArgumentError(
            "conditional model time must lie in 1:$(length(xs) - 1); received $t"
        ),
    )
    return nothing
end

struct ConditionalDynamics{D,X}
    component::D
    trajectory::X
end
function (d::ConditionalDynamics)(ctx)
    t = ctx.t
    _validate_trajectory_time(d.trajectory, t)
    return inner_dynamics(
        d.component,
        t,
        _trajectory_state(d.trajectory, t - 1),
        _trajectory_state(d.trajectory, t),
    )
end

struct ConditionalObservation{O,X}
    component::O
    trajectory::X
end
function (o::ConditionalObservation)(ctx)
    t = ctx.t
    _validate_trajectory_time(o.trajectory, t)
    return inner_observation(o.component, t, _trajectory_state(o.trajectory, t))
end

"""
    condition_inner(model::HierarchicalSSM, outer_trajectory::AbstractVector)

Return an ordinary `StateSpaceModel` describing the inner process conditional on the outer
trajectory. A `ReferenceTrajectory` is indexed `0:T`; an ordinary one-based vector stores
`x0` at index 1 and `x_t` at index `t+1`. At least the initial state is required.

The prior is resolved once. Dynamics and observations resolve lazily, without materialising
per-time matrices, and retain the original component values (including activity flags).
The trajectory is borrowed and must remain unchanged while the conditional model is used:
rebuild the view when trajectory values or model parameters change. No numerical filtering
state or AD workspace is cached. Differentiation through construction can include the
trajectory; holding this view fixed during AD instead treats its captured inputs as fixed.

This operation supplies a model, not an algorithm capability: an appropriate analytical
filter is still required, and backward prediction/ancestor sampling support is separate.
"""
function condition_inner(model::HierarchicalSSM, xs::AbstractVector)
    _validate_trajectory(xs)
    return StateSpaceModel(
        inner_prior(model, _trajectory_state(xs, 0)),
        TimeVaryingDynamics(ConditionalDynamics(model.dyn.inner, xs)),
        TimeVaryingObservation(ConditionalObservation(model.obs.inner, xs)),
    )
end

# A small hook lets ordinary likelihood evaluators check a conditional model's horizon.
_validate_observation_horizon(component, ys) = nothing
function _validate_observation_horizon(
    c::Union{ConditionalDynamics,ConditionalObservation}, ys
)
    length(c.trajectory) == length(ys) + 1 || throw(
        DimensionMismatch(
            "trajectory must contain one initial state plus one state per observation"
        ),
    )
    return nothing
end
_validate_observation_horizon(c::TimeVarying, ys) = _validate_observation_horizon(c.f, ys)
function _validate_observations(model::StateSpaceModel, ys::AbstractVector)
    Base.require_one_based_indexing(ys)
    _validate_observation_horizon(model.dyn, ys)
    _validate_observation_horizon(model.obs, ys)
    return nothing
end

## GENERATIVE INTERFACE ####################################################################

function simulate(rng::AbstractRNG, p::HierarchicalPrior)
    x = simulate(rng, p.outer)
    z = simulate(rng, inner_prior(p, x))
    return HierarchicalState(x, z)
end

function simulate(
    rng::AbstractRNG, d::HierarchicalDynamics, t::Integer, s::HierarchicalState
)
    x = simulate(rng, d.outer, t, s.x)
    z = simulate(rng, inner_dynamics(d, t, s.x, x), t, s.z)
    return HierarchicalState(x, z)
end

function simulate(
    rng::AbstractRNG, o::HierarchicalObservation, t::Integer, s::HierarchicalState
)
    return simulate(rng, inner_observation(o, t, s.x), t, s.z)
end

function logdensity(p::HierarchicalPrior, s::HierarchicalState)
    return logdensity(p.outer, s.x) + logdensity(inner_prior(p, s.x), s.z)
end

function logdensity(
    d::HierarchicalDynamics, t::Integer, sp::HierarchicalState, sn::HierarchicalState
)
    return logdensity(d.outer, t, sp.x, sn.x) +
           logdensity(inner_dynamics(d, t, sp.x, sn.x), t, sp.z, sn.z)
end

function logdensity(o::HierarchicalObservation, t::Integer, s::HierarchicalState, y)
    return logdensity(inner_observation(o, t, s.x), t, s.z, y)
end
