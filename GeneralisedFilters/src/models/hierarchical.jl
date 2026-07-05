export HierarchicalPrior, HierarchicalDynamics, HierarchicalObservation, HierarchicalSSM
export HierarchicalState

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

## GENERATIVE INTERFACE ####################################################################

function simulate(rng::AbstractRNG, p::HierarchicalPrior)
    x = simulate(rng, p.outer)
    z = simulate(rng, resolve(p.inner, (; x0=x)))
    return HierarchicalState(x, z)
end

function simulate(
    rng::AbstractRNG, d::HierarchicalDynamics, t::Integer, s::HierarchicalState
)
    x = simulate(rng, d.outer, t, s.x)
    z = simulate(rng, resolve(d.inner, (; t, x_prev=s.x, x_new=x)), t, s.z)
    return HierarchicalState(x, z)
end

function simulate(
    rng::AbstractRNG, o::HierarchicalObservation, t::Integer, s::HierarchicalState
)
    return simulate(rng, resolve(o.inner, (; t, x=s.x)), t, s.z)
end

function logdensity(p::HierarchicalPrior, s::HierarchicalState)
    return logdensity(p.outer, s.x) + logdensity(resolve(p.inner, (; x0=s.x)), s.z)
end

function logdensity(
    d::HierarchicalDynamics, t::Integer, sp::HierarchicalState, sn::HierarchicalState
)
    return logdensity(d.outer, t, sp.x, sn.x) +
           logdensity(resolve(d.inner, (; t, x_prev=sp.x, x_new=sn.x)), t, sp.z, sn.z)
end

function logdensity(o::HierarchicalObservation, t::Integer, s::HierarchicalState, y)
    return logdensity(resolve(o.inner, (; t, x=s.x)), t, s.z, y)
end
