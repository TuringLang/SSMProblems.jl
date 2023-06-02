"""
A unified interface to define State Space Models interfaces in the context of Particle MCMC algorithms.
"""
module SSMProblems

"""
"""
abstract type AbstractParticle end
abstract type AbstractParticleCache end

"""
    new_particle = transition!!(rng, step, particle, cache)

Simulate the particle for the next time step from the forward dynamics.
"""
function transition!!(
    rng,
    step,
    particle::AbstractParticle,
    cache::AbstractParticleCache = nothing,
) end

"""
    ℓM = transition_logdensity(step, particle, x, cache)

(Optional) Computes the log-density of the forward transition if the density is available.
"""
function transition_logdensity(
    step,
    particle::AbstractParticle,
    x,
    cache::AbstractParticleCache = nothing,
) end

"""
    ℓπ = logdensity(step, particle, cache)

Compute the log potential of current particle. This effectively "reweight" each particle.
"""
function emission_logdensity(
    step,
    particle::AbstractParticle,
    cache::AbstractParticleCache = nothing,
) end

"""
    isdone(step, particle, cache=nothing)

Determine whether we have reached the last time step of the Markov process. Return `true` if yes, otherwise return `false`.
"""
function isdone(step, particle::AbstractParticle, cache::AbstractParticleCache = nothing) end

export transition!!, transition_logdensity, emission_logdensity, isdone, AbstractParticle

end
