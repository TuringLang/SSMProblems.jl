"""
A unified interface to define State Space Models interfaces in the context of Particle MCMC algorithms.
"""
module SSMProblems

"""
"""
abstract type AbstractParticle end
abstract type AbstractParticleCache end

"""
    transition!!(rng, step, particle[, cache])

Simulate the particle for the next time step from the forward dynamics.
"""
function transition!! end

"""
    transition_logdensity(step, particle, x[, cache])

(Optional) Computes the log-density of the forward transition if the density is available.
"""
function transition_logdensity end

"""
    emission_logdensity(step, particle[, cache])

Compute the log potential of current particle. This effectively "reweight" each particle.
"""
function emission_logdensity end

"""
    isdone(step, particle[, cache])

Determine whether we have reached the last time step of the Markov process. Return `true` if yes, otherwise return `false`.
"""
function isdone end

export transition!!, transition_logdensity, emission_logdensity, isdone, AbstractParticle

end
