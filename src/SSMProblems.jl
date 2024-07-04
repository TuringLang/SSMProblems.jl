"""
A unified interface to define state space models in the context of particle MCMC algorithms.
"""
module SSMProblems

using AbstractMCMC: AbstractMCMC
import Random: AbstractRNG

export LatentDynamics, ObservationProcess, AbstractStateSpaceModel, StateSpaceModel

"""
    Latent dynamics of a state space model.

    Any concrete subtype of `LatentDynamics` must implement at least one method from each of
    the following pairs, as documented below:
        - `initialise`/`initialise_logdensity`
        - `transition`/`transition_logdensity`
    
    Alternatively, you may specify methods called `initialisation_distribution` and
    `transition_distribution`, which will be used to define the above pairs, respectively.
"""
abstract type LatentDynamics end

"""
    initialise(
        rng::AbstractRNG, 
        dynamics::LatentDynamics;
        extra
    )

    Simulate the initial state.
"""
function initialise end

"""
    initialisation_logdensity(
        dynamics::LatentDynamics
        state;
        extra
    )

    Compute the log-density of an initial state.
"""
function initialisation_logdensity end

"""
    initialise_distribution(
        dynamics::LatentDynamics;
        extra
    )

    Return the distribution of the initial state.
"""
function initialisation_distribution end

"""
    transition(
        rng::AbstractRNG, 
        dynamics::LatentDynamics;
        state,
        step,
        extra
    )

    Simulate the state at the next time step given the current state.
"""
function transition end

"""
    transition_logdensity(
        dynamics::LatentDynamics
        next_state;
        state,
        step,
        extra
    )

    Compute the log-density of a state at the next time step given the current state.
"""
function transition_logdensity end

"""
    transition_distribution(
        dynamics::LatentDynamics;
        state,
        step,
        extra
    )

    Return the distribution of the state at the next time step given the current state.
"""
function transition_distribution end

"""
    Observation process of a state space model.

    Any concrete subtype of `ObservationProcess` must implement the `observation_logdensity`
    method, as defined below. Optionally, it may also imprement `observation` for use in
    forward simulation of the state space model.
    
    Alternatively, you may specify a method called `observation_distribution`, which will be 
    used to define both of the above methods.
"""
abstract type ObservationProcess end

"""
    observation(
        rng::AbstractRNG, 
        process::ObservationProcess;
        state,
        step,
        extra
    )

    Simulate an observation given the current state.
"""
function observation end

"""
    observation_logdensity(
        process::ObservationProcess;
        observation,
        state,
        step,
        extra
    )

    Compute the log-density of an observation given the current state.
"""

function observation_logdensity end

"""
    observation_distribution(
        process::ObservationProcess;
        state,
        step,
        extra
    )

    Return the distribution of an observation given the current state.
"""

"""
    An abstract type for state space models.

    Any concrete subtype of `AbstractStateSpaceModel` should implement `initialise`,
    `transition`, `observation`, and their corresponding log-density methods.

    For most regular use-cases, the predefined `StateSpaceModel` type, documented below,
    should be sufficient.
"""
abstract type AbstractStateSpaceModel <: AbstractMCMC.AbstractModel end

"""
    A state space model.

    A vanilla implementation of a state space model, composed of a latent dynamics and an
    observation process.
"""
struct StateSpaceModel{LD<:LatentDynamics,OP<:ObservationProcess} <: AbstractStateSpaceModel
    latent_dynamics::LD
    observation_process::OP
end

# SSM-level methods simply call the corresponding methods of the latent dynamics and
# observation process
function initialise(rng::AbstractRNG, model::StateSpaceModel; extra...)
    return initialise(rng, model.latent_dynamics; extra...)
end
function initialisation_logdensity(model::StateSpaceModel; extra...)
    return initialisation_logdensity(model.latent_dynamics; extra...)
end
function initialisation_distribution(model::StateSpaceModel; extra...)
    return initialisation_distribution(model.latent_dynamics; extra...)
end
function transition(rng::AbstractRNG, model::StateSpaceModel; state, step, extra...)
    return transition(rng, model.latent_dynamics; state, step, extra...)
end
function transition_logdensity(model::StateSpaceModel; next_state, state, step, extra...)
    return transition_logdensity(model.latent_dynamics; next_state, state, step, extra...)
end
function transition_distribution(model::StateSpaceModel; state, step, extra...)
    return transition_distribution(model.latent_dynamics; state, step, extra...)
end
function observation(rng::AbstractRNG, model::StateSpaceModel; state, step, extra...)
    return observation(rng, model.observation_process; state, step, extra...)
end
function observation_logdensity(model::StateSpaceModel; observation, state, step, extra...)
    return observation_logdensity(
        model.observation_process; observation, state, step, extra...
    )
end
function observation_distribution(model::StateSpaceModel; state, step, extra...)
    return observation_distribution(model.observation_process; state, step, extra...)
end

include("utils/distributions.jl")
include("utils/forward_simulation.jl")

end
