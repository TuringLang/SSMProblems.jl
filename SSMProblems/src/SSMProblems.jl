"""
A minimal interface for defining state-space models.

This package defines process and model interfaces, distribution adapters, and the
`distribution`/`simulate`/`logdensity` generics. Inference
packages build on these without depending on each other. Conditioning mechanisms, parameter
atoms and algorithm-specific machinery deliberately live downstream.
"""
module SSMProblems

using Distributions: MvNormal, logpdf
using LinearAlgebra: cholesky
using Random: AbstractRNG, default_rng, randn
using StaticArrays: SVector, @SVector
using Statistics: mean, cov

export StatePrior, LatentDynamics, ObservationProcess
export AbstractStateSpaceModel, StateSpaceModel, prior, dyn, obs
export DistributionPrior, DistributionDynamics, DistributionObservation
export distribution, simulate, logdensity, simulate_from_dist

## PROCESSES ###############################################################################

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

## GENERICS ################################################################################

"""
    distribution(prior::StatePrior)
    distribution(dyn::LatentDynamics, t::Integer, x)
    distribution(obs::ObservationProcess, t::Integer, x)

Return the distribution associated with a model component. Implementing this method derives
`simulate` and `logdensity` for free; components may instead implement those directly when
no tractable distribution object is available.

Components take no keyword arguments. Dependence on parameters, controls or an outer state
is expressed by the component object itself, which downstream packages may build per step.
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

"""
    simulate(rng::AbstractRNG, prior::StatePrior)
    simulate(rng::AbstractRNG, dyn::LatentDynamics, t::Integer, x)
    simulate(rng::AbstractRNG, obs::ObservationProcess, t::Integer, x)

Draw from a model component, by default from its `distribution`.
"""
function simulate end

"""
    logdensity(prior::StatePrior, x0)
    logdensity(dyn::LatentDynamics, t::Integer, x_prev, x_new)
    logdensity(obs::ObservationProcess, t::Integer, x, y)

Evaluate the log-density of a model component, by default that of its `distribution`.
"""
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

## DISTRIBUTION ADAPTERS ####################################################################

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

## MODEL ###################################################################################

"""
    AbstractStateSpaceModel

Interface for a state-space model. Subtypes implement [`prior`](@ref), [`dyn`](@ref), and
[`obs`](@ref) to expose their model components. A custom model need not store components in
fields with those names. Inference algorithms may require additional structure from the
returned components.
"""
abstract type AbstractStateSpaceModel end

"""
    prior(model::AbstractStateSpaceModel)

Return the model's initial state prior, a [`StatePrior`](@ref).
"""
function prior end

"""
    dyn(model::AbstractStateSpaceModel)

Return the model's latent dynamics, a [`LatentDynamics`](@ref).
"""
function dyn end

"""
    obs(model::AbstractStateSpaceModel)

Return the model's observation process, an [`ObservationProcess`](@ref).
"""
function obs end

"""
    StateSpaceModel(prior, dyn, obs)
    StateSpaceModel(model::AbstractStateSpaceModel)

A state-space model composed of an initial state prior, latent dynamics, and an observation
process. The one-argument constructor obtains the components through [`prior`](@ref),
[`dyn`](@ref), and [`obs`](@ref). It retains those components without copying them; an
existing `StateSpaceModel` is returned unchanged.
"""
struct StateSpaceModel{P<:StatePrior,D<:LatentDynamics,O<:ObservationProcess} <:
       AbstractStateSpaceModel
    prior::P
    dyn::D
    obs::O
end

function StateSpaceModel(model::AbstractStateSpaceModel)
    return StateSpaceModel(prior(model), dyn(model), obs(model))
end
StateSpaceModel(model::StateSpaceModel) = model

prior(model::StateSpaceModel) = model.prior
dyn(model::StateSpaceModel) = model.dyn
obs(model::StateSpaceModel) = model.obs

## FORWARD SIMULATION ######################################################################

"""
    simulate([rng,] model::AbstractStateSpaceModel, T::Integer)

Simulate a trajectory of length `T`, returning `(x0, xs, ys)` where `x0` is the initial
state, `xs` the states at times `1:T`, and `ys` the observations at times `1:T`.
"""
function simulate(rng::AbstractRNG, model::AbstractStateSpaceModel, T::Integer)
    T >= 0 || throw(ArgumentError("simulation length must be nonnegative"))
    x0 = simulate(rng, prior(model))
    T == 0 && return (x0, typeof(x0)[], Any[])
    xs = fill(simulate(rng, dyn(model), 1, x0), T)
    for t in 2:T
        xs[t] = simulate(rng, dyn(model), t, xs[t - 1])
    end
    ys = map(t -> simulate(rng, obs(model), t, xs[t]), 1:T)
    return x0, xs, ys
end
simulate(model::AbstractStateSpaceModel, T::Integer) = simulate(default_rng(), model, T)

end
