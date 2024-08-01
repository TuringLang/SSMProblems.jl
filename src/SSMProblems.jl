"""
A unified interface to define state space models in the context of particle MCMC algorithms.
"""
module SSMProblems

using AbstractMCMC: AbstractMCMC
import Random: AbstractRNG, default_rng
import Distributions: logpdf

export LatentDynamics, ObservationProcess, AbstractStateSpaceModel, StateSpaceModel

"""
    Latent dynamics of a state space model.

    Any concrete subtype of `LatentDynamics` should implement the functions `logdensity` and
    `simulate`, by defining two methods as documented below, one for initialisation and one
    for transitioning. Whether each of these functions need to be implemented depends on the
    exact inference algorithm that is intended to be used.

    Alternatively, you may specify methods for the function `distribution` which will be
    used to define the above methods.
"""
abstract type LatentDynamics end

"""
    Observation process of a state space model.

    Any concrete subtype of `ObservationProcess` must implement the `logdensity`
    method, as defined below. Optionally, it may also implement `simulate` for use in
    forward simulation of the state space model.
    
    Alternatively, you may specify a method for `distribution`, which will be used to define
    both of the above methods.
"""
abstract type ObservationProcess end

"""
    distribution(dyn::LatentDynamics, extra)

Return the initialisation distribution for the latent dynamics.

The method should return the distribution of the initial state of the latent dynamics.
The  returned value should be a `Distributions.Distribution` object that implements
sampling and log-density calculations.

See also [`LatentDynamics`](@ref).

# Returns
- `Distributions.Distribution`: The distribution of the initial state.
"""
function distribution(dyn::LatentDynamics, extra)
    throw(MethodError(distribution, (dyn, extra)))
end

"""
    distribution(dyn::LatentDynamics, step::Integer, state, extra)

Return the transition distribution for the latent dynamics.

The method should return the distribution of the state for the next time step given the
current state `state` at time step `step`. The returned value should be a
`Distributions.Distribution` object that implements sampling and log-density
calculations. 

See also [`LatentDynamics`](@ref).

# Returns
- `Distributions.Distribution`: The distribution of the new state.
"""
function distribution(dyn::LatentDynamics, step::Integer, state, extra)
    throw(MethodError(distribution, (dyn, step, state, extra)))
end

"""
    distribution(obs::ObservationProcess, step::Integer, state, extra)

Return the observation distribution for the observation process.

The method should return the distribution of an observation given the current state
`state` at time step `step`. The returned value should be a `Distributions.Distribution`
object that implements sampling and log-density calculations.

See also [`ObservationProcess`](@ref).

# Returns
- `Distributions.Distribution`: The distribution of the observation.
"""
function distribution(obs::ObservationProcess, step::Integer, state, extra)
    throw(MethodError(distribution, (obs, step, state, extra)))
end

"""
    simulate([rng::AbstractRNG], dyn::LatentDynamics, extra)

    Simulate an initial state for the latent dynamics.

    The method should return a random initial state for the first time step of the latent
    dynamics.

    The default behaviour is generate a random sample from distribution returned by the
    corresponding `distribution()` method.

    See also [`LatentDynamics`](@ref).
"""
function simulate(rng::AbstractRNG, dyn::LatentDynamics, extra)
    return rand(rng, distribution(dyn, extra))
end
function simulate(dyn::LatentDynamics, extra)
    return simulate(default_rng(), dyn, extra)
end

"""
    simulate([rng::AbstractRNG], dyn::LatentDynamics, step::Integer, state, extra)

Simulate a transition of the latent dynamics.

The method should return a random state for the next time step given the state `state` 
at the current time step, `step`.

The default behaviour is generate a random sample from distribution returned by the
corresponding `distribution()` method.

See also [`LatentDynamics`](@ref).
"""
function simulate(rng::AbstractRNG, dyn::LatentDynamics, step::Integer, state, extra)
    return rand(rng, distribution(dyn, step, state, extra))
end
function simulate(dynamics::LatentDynamics, state, step, extra)
    return simulate(default_rng(), dynamics, state, step, extra)
end

"""
    simulate([rng::AbstractRNG], process::ObservationProcess, step::Integer, state, extra)

Simulate an observation given the current state.

The method should return a random observation given the current state `state` at time
step `step`.

The default behaviour is generate a random sample from distribution returned by the
corresponding `distribution()` method.

See also [`ObservationProcess`](@ref).
"""
function simulate(rng::AbstractRNG, obs::ObservationProcess, step::Integer, state, extra)
    return rand(rng, distribution(obs, step, state, extra))
end
function simulate(obs::ObservationProcess, step::Integer, state, extra)
    return simulate(default_rng(), obs, step, state, extra)
end

"""
    logdensity(dyn::LatentDynamics, new_state, extra)

Compute the log-density of an initial state for the latent dynamics.

The method should return the log-density of the initial state `new_state` for the
initial time step of the latent dynamics.

The default behaviour is to compute the log-density of the distribution return by the
corresponding `distribution()` method.

See also [`LatentDynamics`](@ref).
"""
function logdensity(dyn::LatentDynamics, new_state, extra)
    return logpdf(distribution(dyn, extra), new_state)
end

"""
    logdensity(dyn::LatentDynamics, step::Integer, state, new_state, extra)

Compute the log-density of a transition of the latent dynamics.

The method should return the log-density of the new state `new_state` given the current
state `state` at time step `step`.

The default behaviour is to compute the log-density of the distribution return by the
corresponding `distribution()` method.

See also [`LatentDynamics`](@ref).
"""
function logdensity(dyn::LatentDynamics, step::Integer, state, new_state, extra)
    return logpdf(distribution(dyn, step, state, extra), new_state)
end

"""
    logdensity(obs::ObservationProcess, step::Integer, state, observation, extra)

Compute the log-density of an observation given the current state.

The method should return the log-density of the observation `observation` given the
current state `state` at time step `step`.

The default behaviour is to compute the log-density of the distribution return by the
corresponding `distribution()` method.

See also [`ObservationProcess`](@ref).
"""
function logdensity(obs::ObservationProcess, step::Integer, state, observation, extra)
    return logpdf(distribution(obs, step, state, extra), observation)
end

"""
    An abstract type for state space models.

    Any concrete subtype of `AbstractStateSpaceModel` should implement a method for
    `AbstractMCMC.sample` which performs forward simulation. For an example implementation,
    see [AbstractMCMC.sample(::StateSpaceModel)](@ref).

    For most regular use-cases, the predefined `StateSpaceModel` type, documented below,
    should be sufficient.
"""
abstract type AbstractStateSpaceModel <: AbstractMCMC.AbstractModel end

"""
    A state space model.

    A vanilla implementation of a state space model, composed of a latent dynamics and an
    observation process.

    # Fields
    - `dyn::LD`: The latent dynamics of the state space model.
    - `obs::OP`: The observation process of the state space model.
"""
struct StateSpaceModel{LD<:LatentDynamics,OP<:ObservationProcess} <: AbstractStateSpaceModel
    dyn::LD
    obs::OP
end

include("utils/forward_simulation.jl")

end
