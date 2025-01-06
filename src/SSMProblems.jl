"""
A unified interface to define state space models in the context of particle MCMC algorithms.
"""
module SSMProblems

using AbstractMCMC: AbstractMCMC
import Base: eltype
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

    All of these methods should accept keyword arguments through `kwargs...` to facilitate
    inference-time dependencies of the dynamics as explained in [Control Variables and Keyword Arguments](@ref).

    The latent states should be of type `ET` which should be a composed from `T`, the
    arithmetic type used for the dynamics (e.g. Float32, ForwardDiff.Dual).

    # Parameters
    - `T`: The arithmetic type of the latent dynamics.
    - `ET`: The element type of the latent dynamics.
"""
abstract type LatentDynamics{T<:Real,ET} end

"""
    arithmetic_type(::Type{<:LatentDynamics})
    arithmetic_type(dyn::LatentDynamics)

    Return the arithmetic type of the latent dynamics.
"""
arithmetic_type(::Type{<:LatentDynamics{T}}) where {T} = T
arithmetic_type(dyn::LatentDynamics) = arithmetic_type(typeof(dyn))

"""
    eltype(::Type{<:LatentDynamics})
    eltype(dyn::LatentDynamics)

    Return the type of the state of the latent dynamics.
"""
Base.eltype(::Type{<:LatentDynamics{T,ET}}) where {T,ET} = ET
Base.eltype(dyn::LatentDynamics) = eltype(typeof(dyn))

"""
    Observation process of a state space model.

    Any concrete subtype of `ObservationProcess` must implement the `logdensity`
    method, as defined below. Optionally, it may also implement `simulate` for use in
    forward simulation of the state space model.
    
    Alternatively, you may specify a method for `distribution`, which will be used to define
    both of the above methods.

    All of these methods should accept keyword arguments through `kwargs...` to facilitate
    inference-time dependencies of the observations as explained in [Control Variables and Keyword Arguments](@ref).

    The observations should be of type `ET` which should be a composed from `T`, the
    arithmetic type used for the observations (e.g. Float32, ForwardDiff.Dual).

    # Parameters
    - `T`: The arithmetic type of the observation process.
    - `ET`: The element type of the observation process.
"""
abstract type ObservationProcess{T<:Real,ET} end

"""
    arithmetic_type(::Type{<:ObservationProcess})
    arithmetic_type(obs::ObservationProcess)

    Return the arithmetic type of the observation process.
"""
arithmetic_type(::Type{<:ObservationProcess{T}}) where {T} = T
arithmetic_type(obs::ObservationProcess) = arithmetic_type(typeof(obs))

"""
    eltype(::Type{<:ObservationProcess})
    eltype(obs::ObservationProcess)

    Return the type of the observations of the observation process.
"""
Base.eltype(::Type{<:ObservationProcess{T,ET}}) where {T,ET} = ET
Base.eltype(obs::ObservationProcess) = eltype(typeof(obs))

"""
    distribution(dyn::LatentDynamics; kwargs...)

Return the initialisation distribution for the latent dynamics.

The method should return the distribution of the initial state of the latent dynamics.
The  returned value should be a `Distributions.Distribution` object that implements
sampling and log-density calculations.

See also [`LatentDynamics`](@ref).

# Returns
- `Distributions.Distribution`: The distribution of the initial state.
"""
function distribution(dyn::LatentDynamics; kwargs...)
    throw(MethodError(distribution, (dyn)))
end

"""
    distribution(dyn::LatentDynamics, step::Integer, prev_state; kwargs...)

Return the transition distribution for the latent dynamics.

The method should return the distribution of the current state (at time step `step`) given 
the previous state `prev_state`. The returned value should be a `Distributions.Distribution`
object that implements sampling and log-density calculations. 

See also [`LatentDynamics`](@ref).

# Returns
- `Distributions.Distribution`: The distribution of the new state.
"""
function distribution(dyn::LatentDynamics, step::Integer, state; kwargs...)
    throw(MethodError(distribution, (dyn, step, state, kwargs...)))
end

"""
    distribution(obs::ObservationProcess, step::Integer, state; kwargs...)

Return the observation distribution for the observation process.

The method should return the distribution of an observation given the current state
`state` at time step `step`. The returned value should be a `Distributions.Distribution`
object that implements sampling and log-density calculations.

See also [`ObservationProcess`](@ref).

# Returns
- `Distributions.Distribution`: The distribution of the observation.
"""
function distribution(obs::ObservationProcess, step::Integer, state; kwargs...)
    throw(MethodError(distribution, (obs, step, state, kwargs...)))
end

"""
    simulate([rng::AbstractRNG], dyn::LatentDynamics; kwargs...)

    Simulate an initial state for the latent dynamics.

    The method should return a random initial state for the first time step of the latent
    dynamics.

    The default behaviour is generate a random sample from distribution returned by the
    corresponding `distribution()` method.

    See also [`LatentDynamics`](@ref).
"""
function simulate(rng::AbstractRNG, dyn::LatentDynamics; kwargs...)
    return rand(rng, distribution(dyn; kwargs...))
end
simulate(dyn::LatentDynamics; kwargs...) = simulate(default_rng(), dyn; kwargs...)

"""
    simulate([rng::AbstractRNG], dyn::LatentDynamics, step::Integer, prev_state; kwargs...)

Simulate a transition of the latent dynamics.

The method should return a random state for the current time step, `step`,  given the
previous state, `prev_state`.

The default behaviour is generate a random sample from distribution returned by the
corresponding `distribution()` method.

See also [`LatentDynamics`](@ref).
"""
function simulate(
    rng::AbstractRNG, dyn::LatentDynamics, step::Integer, prev_state; kwargs...
)
    return rand(rng, distribution(dyn, step, prev_state; kwargs...))
end
function simulate(dynamics::LatentDynamics, prev_state, step; kwargs...)
    return simulate(default_rng(), dynamics, prev_state, step; kwargs...)
end

"""
    simulate([rng::AbstractRNG], process::ObservationProcess, step::Integer, state; kwargs...)

Simulate an observation given the current state.

The method should return a random observation given the current state `state` at time
step `step`.

The default behaviour is generate a random sample from distribution returned by the
corresponding `distribution()` method.

See also [`ObservationProcess`](@ref).
"""
function simulate(
    rng::AbstractRNG, obs::ObservationProcess, step::Integer, state; kwargs...
)
    return rand(rng, distribution(obs, step, state; kwargs...))
end
function simulate(obs::ObservationProcess, step::Integer, state; kwargs...)
    return simulate(default_rng(), obs, step, state; kwargs...)
end

"""
    logdensity(dyn::LatentDynamics, new_state; kwargs...)

Compute the log-density of an initial state for the latent dynamics.

The method should return the log-density of the initial state `new_state` for the
initial time step of the latent dynamics.

The default behaviour is to compute the log-density of the distribution return by the
corresponding `distribution()` method.

See also [`LatentDynamics`](@ref).
"""
function logdensity(dyn::LatentDynamics, new_state; kwargs...)
    return logpdf(distribution(dyn; kwargs...), new_state)
end

"""
    logdensity(dyn::LatentDynamics, step::Integer, prev_state, new_state; kwargs...)

Compute the log-density of a transition of the latent dynamics.

The method should return the log-density of the new state `new_state` (at time step `step`)
given the previous state `prev_state` 

The default behaviour is to compute the log-density of the distribution return by the
corresponding `distribution()` method.

See also [`LatentDynamics`](@ref).
"""
function logdensity(dyn::LatentDynamics, step::Integer, prev_state, new_state; kwargs...)
    return logpdf(distribution(dyn, step, prev_state; kwargs...), new_state)
end

"""
    logdensity(obs::ObservationProcess, step::Integer, state, observation; kwargs...)

Compute the log-density of an observation given the current state.

The method should return the log-density of the observation `observation` given the
current state `state` at time step `step`.

The default behaviour is to compute the log-density of the distribution return by the
corresponding `distribution()` method.

See also [`ObservationProcess`](@ref).
"""
function logdensity(obs::ObservationProcess, step::Integer, state, observation; kwargs...)
    return logpdf(distribution(obs, step, state; kwargs...), observation)
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

    # Parameters
    - `T`: The arithmetic type of the state space model, which the latent dynamics and
           observation process should be consistent with.
    - `LD`: The type of the latent dynamics.
    - `OP`: The type of the observation process.
"""
struct StateSpaceModel{T<:Real,LD<:LatentDynamics{T},OP<:ObservationProcess{T}} <:
       AbstractStateSpaceModel
    dyn::LD
    obs::OP
    function StateSpaceModel(dyn::LatentDynamics{T}, obs::ObservationProcess{T}) where {T}
        return new{T,typeof(dyn),typeof(obs)}(dyn, obs)
    end
end

"""
    arithmetic_type(::Type{<:StateSpaceModel})
    arithmetic_type(model::StateSpaceModel)

    Return the arithmetic type of the state space model.
"""
arithmetic_type(model::StateSpaceModel) = arithmetic_type(typeof(model))
arithmetic_type(::Type{<:StateSpaceModel{T}}) where {T} = T

include("batch_methods.jl")
include("utils/forward_simulation.jl")

end
