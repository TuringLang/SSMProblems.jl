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
    transition!!(rng, model[, state, timestep, cache])

Simulate the particle for the next time step from the forward dynamics.
"""
function transition!! end

"""
    transition_logdensity(model, prev_state, current_state[, timestep, cache])

(Optional) Computes the log-density of the forward transition if the density is available.
"""
function transition_logdensity end

"""
    emission_logdensity(model, state, observation[, timestep, cache])

Compute the log potential of the current particle. This effectively "reweight" each particle.
"""
function emission_logdensity end

# Include utils and adjacent code
include("utils/particles.jl")

export AbstractStateSpaceModel

end
