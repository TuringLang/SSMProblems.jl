"""
A unified interface to define State Space Models interfaces in the context of Particle MCMC algorithms.
"""
module SSMProblems

"""
    AbstractStateSpaceModel
"""
abstract type AbstractStateSpaceModel end
abstract type AbstractParticleCache end

"""
    transition!!(rng, model[, timestep, state, cache])

Simulate the particle for the next time step from the forward dynamics.
"""
function transition!! end

"""
    transition_logdensity(model, timestep, prev_state, next_state[, cache])

(Optional) Computes the log-density of the forward transition if the density is available.
"""
function transition_logdensity end

"""
    emission_logdensity(model, timestep, state, observation[, cache])

Compute the log potential of the current particle. This effectively "reweight" each particle.
"""
function emission_logdensity end

export AbstractStateSpaceModel

end
