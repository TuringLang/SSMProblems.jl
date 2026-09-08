export StatePrior, LatentDynamics, ObservationProcess
export StateSpaceModel
export distribution, simulate, logdensity, simulate_from_dist
export TimeVaryingDynamics, TimeVaryingObservation
export DistributionPrior, DistributionDynamics, DistributionObservation

"""
    StatePrior

Initial state distribution of a state-space model. A concrete subtype should implement
`simulate` and/or `logdensity`, or supply a `distribution(prior)` method from which both
are derived. Which methods are required depends on the inference algorithm.
"""
abstract type StatePrior end

"""
    LatentDynamics

Transition dynamics of a state-space model. A concrete subtype should implement `simulate`
and/or `logdensity`, or supply a `distribution(dyn, t, x)` method from which both are
derived. Which methods are required depends on the inference algorithm (e.g. a bootstrap
filter needs only `simulate`, a guided proposal also needs `logdensity`).
"""
abstract type LatentDynamics end

"""
    ObservationProcess

Emission process of a state-space model. A concrete subtype should implement `logdensity`
(and optionally `simulate` for forward simulation), or supply a `distribution(obs, t, x)`
method from which both are derived.
"""
abstract type ObservationProcess end

"""
    distribution(prior::StatePrior)
    distribution(dyn::LatentDynamics, t::Integer, x)
    distribution(obs::ObservationProcess, t::Integer, x)

Return the distribution associated with a model component. Implementing this method derives
`simulate` and `logdensity` for free; components may instead implement those directly when
no tractable distribution object is available.
"""
function distribution end

"""
    simulate_from_dist(rng::AbstractRNG, d)

Draw a sample from a distribution object. Defaults to `rand(rng, d)`; specialise it to
return static arrays or otherwise control the sample type.
"""
simulate_from_dist(rng::AbstractRNG, d) = rand(rng, d)

# Preserve static-array types when sampling from an MvNormal with a static mean.
function simulate_from_dist(rng::AbstractRNG, d::MvNormal{T,S,SVector{D,T}}) where {T,S,D}
    z = @SVector randn(rng, T, D)
    return mean(d) + cholesky(cov(d)).L * z
end

function simulate end
function logdensity end

simulate(rng::AbstractRNG, prior::StatePrior) = simulate_from_dist(rng, distribution(prior))
function simulate(rng::AbstractRNG, dyn::LatentDynamics, t::Integer, x)
    return simulate_from_dist(rng, distribution(dyn, t, x))
end
function simulate(rng::AbstractRNG, obs::ObservationProcess, t::Integer, x)
    return simulate_from_dist(rng, distribution(obs, t, x))
end

logdensity(prior::StatePrior, x0) = logpdf(distribution(prior), x0)
function logdensity(dyn::LatentDynamics, t::Integer, x_prev, x_new)
    return logpdf(distribution(dyn, t, x_prev), x_new)
end
function logdensity(obs::ObservationProcess, t::Integer, x, y)
    return logpdf(distribution(obs, t, x), y)
end

"""
    StateSpaceModel(prior, dyn, obs)

A state-space model composed of an initial state prior, latent dynamics, and an observation
process.
"""
struct StateSpaceModel{P<:StatePrior,D<:LatentDynamics,O<:ObservationProcess}
    prior::P
    dyn::D
    obs::O
end

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

resolve(w::TimeVarying, ctx) = w.f(ctx)
distribution(w::TimeVarying, t::Integer, x) = distribution(resolve(w, (; t)), t, x)
function simulate(rng::AbstractRNG, w::TimeVarying, t::Integer, x)
    return simulate(rng, resolve(w, (; t)), t, x)
end
logdensity(w::TimeVarying, t::Integer, a, b) = logdensity(resolve(w, (; t)), t, a, b)

## DISTRIBUTION-RETURNING PROCESS WRAPPERS #################################################

"""
    DistributionPrior(dist)

Lift a distribution object into a `StatePrior`.
"""
struct DistributionPrior{D} <: StatePrior
    dist::D
end

"""
    DistributionDynamics(f)

Lift a closure `f(t, x) -> distribution` into a `LatentDynamics`.
"""
struct DistributionDynamics{F} <: LatentDynamics
    f::F
end

"""
    DistributionObservation(f)

Lift a closure `f(t, x) -> distribution` into an `ObservationProcess`.
"""
struct DistributionObservation{F} <: ObservationProcess
    f::F
end

distribution(p::DistributionPrior) = p.dist
distribution(d::DistributionDynamics, t::Integer, x) = d.f(t, x)
distribution(o::DistributionObservation, t::Integer, x) = o.f(t, x)

## FORWARD SIMULATION ######################################################################

"""
    simulate([rng,] model::StateSpaceModel, T::Integer)

Simulate a trajectory of length `T`, returning `(x0, xs, ys)` where `x0` is the initial
state, `xs` the states at times `1:T`, and `ys` the observations at times `1:T`.
"""
function simulate(rng::AbstractRNG, model::StateSpaceModel, T::Integer)
    T >= 0 || throw(ArgumentError("simulation length must be nonnegative"))
    x0 = simulate(rng, model.prior)
    T == 0 && return (x0, typeof(x0)[], Any[])
    xs = fill(simulate(rng, model.dyn, 1, x0), T)
    for t in 2:T
        xs[t] = simulate(rng, model.dyn, t, xs[t - 1])
    end
    ys = map(t -> simulate(rng, model.obs, t, xs[t]), 1:T)
    return x0, xs, ys
end
simulate(model::StateSpaceModel, T::Integer) = simulate(default_rng(), model, T)
