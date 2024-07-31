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
    for transitioting. Whether each of these functions need to be implemented depends on the
    exact inference algorithm that is intended to be used.

    Alternatively, you may specify methods for the function `distribution` which will be
    used to define the above methods.
"""
abstract type LatentDynamics end

"""
    simulate(
        rng::AbstractRNG, 
        dynamics::LatentDynamics,
        [step::Integer,
        state,]
        extra
    )

    Simulate a new state from the latent dynamics.

    The method containing the `step` and `state` parameters is used for transitioning the
    latent dynamics. Specifically, it should return the state for the next time step given
    the state at the current time step, `step`.

    The method without the `step` and `state` parameters is used for randomly initialising
    the latent dynamics. That is, it should return the state at the first time step.

    See also [`LatentDynamics`](@ref).
"""
function simulate end

"""
    logdensity(
        dynamics::LatentDynamics,
        [step::Integer,
        state,]
        new_state,
        extra
    )

    Compute the log-density of a new state according to the latent dynamics.

    The method containing the `step` and `state` parameters is used for calculating the
    transition log-density. Specifically, it should return the log-density of new state
    `new_state` given the current state `state` at time step `step`.

    The method without the `step` and `state` parameters is used for calculating the initial
    log-density. That is, it should return the log-density of the initial state, contained
    in the argument `new_state`.

    See also [`LatentDynamics`](@ref).
"""
function logdensity end

"""
    distribution(
        dynamics::LatentDynamics,
        [step::Integer,
        state,]
        extra
    )

    Return the distribution of a new state according to the latent dynamics.

    The method containing the `step` and `state` parameters returns the transition
    distribution. Specifically, it should return the distribution of the new state given
    the current state `state` at time step `step`. The returned value should be a
    `Distributions.Distribution` object that implements sampling and log-density
    calculations. 

    See also [`LatentDynamics`](@ref).

    # Returns
    - `Distributions.Distribution`: The distribution of the new state.
"""
function distribution end

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
    simulate(
        rng::AbstractRNG, 
        process::ObservationProcess,
        step,
        state,
        extra
    )

    Simulate an observation given the current state.

    See also [`ObservationProcess`](@ref).
"""
function simulate end

"""
    logdensity(
        process::ObservationProcess,
        step,
        state,
        observation,
        extra
    )

    Compute the log-density of an observation given the current state.

    See also [`ObservationProcess`](@ref).
"""
function logdensity end

"""
    distribution(
        process::ObservationProcess,
        step,
        state,
        extra
    )

    Return the distribution of an observation given the current state.

    The returned value should be a `Distributions.Distribution` object that implements
    sampling and log-density calculations.

    See also [`ObservationProcess`](@ref).

    # Returns
    - `Distributions.Distribution`: The distribution of the observation.
"""
function distribution end

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

include("utils/distributions.jl")
include("utils/forward_simulation.jl")

end
