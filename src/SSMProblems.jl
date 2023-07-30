"""
A unified interface to define State Space Models interfaces in the context of Particle MCMC algorithms.
"""
module SSMProblems

"""
    AbstractStateSpaceModel
"""
abstract type AbstractStateSpaceModel end

"""
    AbstractParticle{T<:AbstractStateSpaceModel}
"""
abstract type AbstractParticle{T<:AbstractStateSpaceModel} end
abstract type AbstractParticleCache end

"""
    transition!!(rng, step, model, particle[, cache])

Simulate the particle for the next time step from the forward dynamics.
"""
function transition!! end

"""
    transition_logdensity(step, model, prev_particle, next_particle[, cache])

(Optional) Computes the log-density of the forward transition if the density is available.
"""
function transition_logdensity end

"""
    emission_logdensity(step, model, particle[, cache])

Compute the log potential of current particle. This effectively "reweight" each particle.
"""
function emission_logdensity end

"""
    isdone(step, model, particle[, cache])

Determine whether we have reached the last time step of the Markov process. Return `true` if yes, otherwise return `false`.
"""
function isdone end

"""
    particleof(::Type{AbstractStateSpaceModel})

Returns the type of the latent state.
"""
particleof(::Type{AbstractStateSpaceModel}) = Nothing
particleof(model::AbstractStateSpaceModel) = particleof(typeof(model))

"""
    dimension(::Type{AbstractStateSpaceModel})

Returns the dimension of the latent state.
"""
dimension(::Type{AbstractStateSpaceModel}) = Nothing
dimension(model::AbstractStateSpaceModel) = dimension(typeof(model))

"""
    latent_space_dimension(::Type{AbstractStateSpaceModel})

Returns the type of the latent space and its dimension.
"""
latent_space_dimension(T::Type{AbstractStateSpaceModel}) = particleof(T), dimension(T)
latent_space_dimension(model::AbstractStateSpaceModel) = latent_space_dimension(typeof(model))

export transition!!,
    transition_logdensity,
    emission_logdensity,
    isdone,
    AbstractParticle,
    AbstractStateSpaceModel,
    dimension,
    particleof,
    latent_space_dimension

end
