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
abstract type Particle end

"""
    new_particle = M!!(rng, t, particle, cache)

Simulate the particle for the next time step from the forward dynamics.
"""
function M!!(rng, t, particle::Particle, cache) end

"""
    ℓM = logM(t, particle, x, cache)

(Optional) Computes the log-density of the forward transition if the density is available.
"""
function logM(t, particle::Particle, x, cache=nothing) end

"""
    ℓπ = logdensity(t, particle, cache)

Compute the log potential of current particle. This effectively "reweight" each particle.
"""
function logdensity(t, particle::Particle, cache) end

"""
    isdone(t, particle, cache=nothing)

Determine whether we have reached the last time step of the Markov process. Return `true` if yes, otherwise return `false`.
"""
function isdone(t, particle::Particle, cache=nothing) end

end

