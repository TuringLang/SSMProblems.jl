module SSMProblems 

"""
    new_particle = M!!(rng, t, particle, cache)

Simulate the particle for the next time step from the forward dynamics.
"""
function M!!(rng, t, particle, cache)
end


"""
    ℓπ = logdensity(t, particle, cache)

Compute the log potential of current particle. This effectively "reweight" each particle.
"""
function logdensity(t, particle, cache)
end


"""
    SSMProblem
"""

## LogDensityProblem convention
struct SSMProblemExample end

M!!(s::SSMProblemExample, args...)
logdensity(s::SSMProblemExample, args...)
get_particletype(s::SSMProblemExample, args...) = Nothing
get_cachetype(s::SSMProblemExample, args...) = Nothing


## new convention 
# SSMProblem(M!!, logdensity, n_particles, ParticleType, cache)

end