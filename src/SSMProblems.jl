module SSMProblems

"""
# LogDensityProblem convention Example

struct SSMProblemExample end

M!!(s::SSMProblemExample, args...) = Nothing
logdensity(s::SSMProblemExample, args...) = Nothing
get_particletype(s::SSMProblemExample, args...) = Nothing
get_cachetype(s::SSMProblemExample, args...) = Nothing

# New convention example. This example might be useful for AdvancedPS.
SSMProblem(M!!, logdensity, n_particles, ParticleType, cache)
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
