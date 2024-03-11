"""
A unified interface to define State Space Models interfaces in the context of Particle MCMC algorithms.
"""
module SSMProblems

using AbstractMCMC: AbstractMCMC

"""
    AbstractStateSpaceModel
"""
abstract type AbstractStateSpaceModel <: AbstractMCMC.AbstractModel end
abstract type AbstractParticleCache end

"""
    transition!!(rng, model[, state, timestep, control, cache])

Simulate the particle for the next time step from the forward dynamics.

# Arguments
- `rng`: random number generator
- `model`: the state space model of interest
- `state`: the current state
- `timestep`: the current time step
- `control`: the control input
- `cache`: the cache object

# Returns
- `next_state`: the next state
"""
function transition!! end

"""
    transition_logdensity(model, prev_state, current_state[, timestep, control, cache])

(Optional) Computes the log-density of the forward transition if the density is available.

# Arguments
- `model`: the state space model of interest
- `prev_state`: the previous state
- `current_state`: the current state
- `timestep`: the current time step
- `control`: the control input
- `cache`: the cache object

# Returns
- `logdensity`: the log-density of the transition
"""
function transition_logdensity end

"""
    emission_logdensity(model, state, observation[, timestep, control, cache])

Compute the log potential of the current particle. This effectively "reweight" each particle.

# Arguments
- `model`: the state space model of interest
- `state`: the current state
- `observation`: the current observation
- `timestep`: the current time step
- `control`: the control input
- `cache`: the cache object

# Returns
- `logdensity`: the log-density of the emission
"""
function emission_logdensity end

# Include utils and adjacent code
include("utils/particles.jl")

export AbstractStateSpaceModel

end
