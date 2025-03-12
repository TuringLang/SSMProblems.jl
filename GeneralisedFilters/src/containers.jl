"""Containers used for storing representations of the filtering distribution."""

## PARTICLES ###############################################################################

"""
    ParticleDistribution

A container for particle filters which composes the weighted sample into a distibution-like
object, with the states (or particles) distributed accoring to their log-weights.
"""
mutable struct ParticleDistribution{PT,WT<:Real}
    particles::Vector{PT}
    ancestors::Vector{Int}
    log_weights::Vector{WT}
end
function ParticleDistribution(particles::Vector{PT}, log_weights::Vector{WT}) where {PT,WT}
    N = length(particles)
    return ParticleDistribution(particles, Vector{Int}(1:N), log_weights)
end

StatsBase.weights(state::ParticleDistribution) = softmax(state.log_weights)

Base.collect(state::ParticleDistribution) = state.particles
Base.length(state::ParticleDistribution) = length(state.particles)
Base.keys(state::ParticleDistribution) = LinearIndices(state.particles)

# not sure if this is kosher, since it doesn't follow the convention of Base.getindex
Base.@propagate_inbounds Base.getindex(state::ParticleDistribution, i) = state.particles[i]
# Base.@propagate_inbounds Base.getindex(state::ParticleDistribution, i::Vector{Int}) = state.particles[i]

function reset_weights!(state::ParticleDistribution{T,WT}) where {T,WT<:Real}
    fill!(state.log_weights, zero(WT))
    return state.log_weights
end

function update_ref!(
    proposed::ParticleDistribution,
    ref_state::Union{Nothing,AbstractVector},
    step::Integer=0,
)
    if !isnothing(ref_state)
        proposed.particles[1] = ref_state[step]
    end
    return proposed
end

## RAO-BLACKWELLISED PARTICLE ##############################################################

"""
    RaoBlackwellisedParticle

A container for Rao-Blackwellised states, composed of a marginalised state `z` (e.g. a
Gaussian or Categorical distribution) and a singular state `x`.
"""
mutable struct RaoBlackwellisedParticle{XT,ZT}
    x::XT
    z::ZT
end

## RAO-BLACKWELLISED PARTICLE DISTRIBUTIONS ################################################

mutable struct BatchRaoBlackwellisedParticles{XT,ZT}
    xs::XT
    zs::ZT
end

mutable struct RaoBlackwellisedParticleDistribution{
    T,M<:CUDA.AbstractMemory,PT<:BatchRaoBlackwellisedParticles
}
    particles::PT
    ancestors::CuVector{Int,M}
    log_weights::CuVector{T,M}
end
function RaoBlackwellisedParticleDistribution(
    particles::PT, log_weights::CuVector{T,M}
) where {T,M,PT}
    N = length(particles)
    return RaoBlackwellisedParticleDistribution(particles, CuVector{Int}(1:N), log_weights)
end

function StatsBase.weights(state::RaoBlackwellisedParticleDistribution)
    return softmax(state.log_weights)
end
function Base.length(state::RaoBlackwellisedParticleDistribution)
    return length(state.log_weights)
end

# Allow particle to be get and set via tree_states[:, 1:N] = states
function Base.getindex(state::BatchRaoBlackwellisedParticles, i)
    return BatchRaoBlackwellisedParticles(state.xs[:, [i]], state.zs[i])
end
function Base.getindex(state::BatchRaoBlackwellisedParticles, i::AbstractVector)
    return BatchRaoBlackwellisedParticles(state.xs[:, i], state.zs[i])
end
function Base.setindex!(
    state::BatchRaoBlackwellisedParticles, value::BatchRaoBlackwellisedParticles, i
)
    state.xs[:, i] = value.xs
    state.zs[i] = value.zs
    return state
end
Base.length(state::BatchRaoBlackwellisedParticles) = size(state.xs, 2)

function expand(particles::CuArray{T,2,Mem}, M::Integer) where {T,Mem<:CUDA.AbstractMemory}
    new_particles = CuArray(zeros(eltype(particles), size(particles, 1), M))
    new_particles[:, 1:size(particles, 2)] = particles
    return new_particles
end

# Method for increasing size of particle container
function expand(p::BatchRaoBlackwellisedParticles, M::Integer)
    new_x = expand(p.xs, M)
    new_z = expand(p.zs, M)
    return BatchRaoBlackwellisedParticles(new_x, new_z)
end

function update_ref!(
    proposed::RaoBlackwellisedParticleDistribution,
    ref_state::Union{Nothing,AbstractVector},
    step::Integer=0,
)
    if !isnothing(ref_state)
        CUDA.@allowscalar begin
            proposed.particles.xs[:, 1] = ref_state[step].xs
            proposed.particles.zs[1] = ref_state[step].zs
        end
    end
    return proposed
end

## BATCH GAUSSIAN DISTRIBUTION #############################################################

mutable struct BatchGaussianDistribution{T}
    μs::CuArray{T,2,CUDA.DeviceMemory}
    Σs::CuArray{T,3,CUDA.DeviceMemory}
end

function Base.getindex(d::BatchGaussianDistribution, i)
    return BatchGaussianDistribution(d.μs[:, [i]], d.Σs[:, :, [i]])
end

function Base.getindex(d::BatchGaussianDistribution, i::AbstractVector)
    return BatchGaussianDistribution(d.μs[:, i], d.Σs[:, :, i])
end

function Base.setindex!(d::BatchGaussianDistribution, value::BatchGaussianDistribution, i)
    d.μs[:, i] = value.μs
    d.Σs[:, :, i] = value.Σs
    return d
end

function expand(d::BatchGaussianDistribution{T}, M::Integer) where {T}
    new_μs = CuArray{T}(undef, size(d.μs, 1), M)
    new_Σs = CuArray{T}(undef, size(d.Σs, 1), size(d.Σs, 2), M)
    new_μs[:, 1:size(d.μs, 2)] = d.μs
    new_Σs[:, :, 1:size(d.Σs, 3)] = d.Σs
    return BatchGaussianDistribution(new_μs, new_Σs)
end
