"""Containers used for storing representations of the filtering distribution."""

## INTERMEDIATES ###########################################################################

mutable struct ParticleIntermediate{DT,AT}
    proposed::DT
    filtered::DT
    ancestors::AT
end

mutable struct Intermediate{DT}
    proposed::DT
    filtered::DT
end

## GAUSSIAN STATES #########################################################################

mutable struct GaussianContainer{XT,ΣT}
    proposed::Gaussian{XT,ΣT}
    filtered::Gaussian{XT,ΣT}
end

mutable struct BatchGaussianDistribution{T,M<:CUDA.AbstractMemory}
    μs::CuArray{T,2,M}
    Σs::CuArray{T,3,M}
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
function expand(d::BatchGaussianDistribution, M::Integer)
    new_μs = CuArray(zeros(eltype(d.μs), size(d.μs, 1), M))
    new_Σs = CuArray(zeros(eltype(d.Σs), size(d.Σs, 1), size(d.Σs, 2), M))
    new_μs[:, 1:size(d.μs, 2)] = d.μs
    new_Σs[:, :, 1:size(d.Σs, 3)] = d.Σs
    return BatchGaussianDistribution(new_μs, new_Σs)
end

## RAO-BLACKWELLISED STATES ################################################################

"""
    RaoBlackwellisedContainer

A container for Rao-Blackwellised states, composed of a marginalised state `z` (e.g. a
Gaussian or Categorical distribution) and a singular state `x`.
"""
mutable struct RaoBlackwellisedContainer{XT,ZT}
    x::XT
    z::ZT
end

mutable struct RaoBlackwellisedParticle{XT,ZT}
    x_particles::XT
    z_particles::ZT
end

# TODO: this needs to be generalised to account for the flatten Levy SSM state
mutable struct RaoBlackwellisedParticleState{
    T,M<:CUDA.AbstractMemory,PT<:RaoBlackwellisedParticle
}
    particles::PT
    log_weights::CuArray{T,1,M}
end

StatsBase.weights(state::RaoBlackwellisedParticleState) = softmax(state.log_weights)
Base.length(state::RaoBlackwellisedParticleState) = length(state.log_weights)

# Allow particle to be get and set via tree_states[:, 1:N] = states
function Base.getindex(state::RaoBlackwellisedParticle, i)
    return RaoBlackwellisedParticle(state.x_particles[:, [i]], state.z_particles[i])
end
function Base.getindex(state::RaoBlackwellisedParticle, i::AbstractVector)
    return RaoBlackwellisedParticle(state.x_particles[:, i], state.z_particles[i])
end
function Base.setindex!(state::RaoBlackwellisedParticle, value::RaoBlackwellisedParticle, i)
    state.x_particles[:, i] = value.x_particles
    state.z_particles[i] = value.z_particles
    return state
end
Base.length(state::RaoBlackwellisedParticle) = size(state.x_particles, 2)

function expand(particles::CuArray{T,2,Mem}, M::Integer) where {T,Mem<:CUDA.AbstractMemory}
    new_particles = CuArray(zeros(eltype(particles), size(particles, 1), M))
    new_particles[:, 1:size(particles, 2)] = particles
    return new_particles
end

# Method for increasing size of particle container
function expand(p::RaoBlackwellisedParticle, M::Integer)
    new_x = expand(p.x_particles, M)
    new_z = expand(p.z_particles, M)
    return RaoBlackwellisedParticle(new_x, new_z)
end

"""
    RaoBlackwellisedParticleContainer
"""
mutable struct RaoBlackwellisedParticleContainer{T,M<:CUDA.AbstractMemory,PT}
    filtered::RaoBlackwellisedParticleState{T,M,PT}
    proposed::RaoBlackwellisedParticleState{T,M,PT}
    ancestors::CuArray{Int,1,M}

    function RaoBlackwellisedParticleContainer(
        x_particles::CuArray{T,2,M}, z_particles::ZT, log_weights::CuArray{T,1,M}
    ) where {T,M<:CUDA.AbstractMemory,ZT}
        init_particles = RaoBlackwellisedParticleState(
            RaoBlackwellisedParticle(x_particles, z_particles), log_weights
        )
        prop_particles = RaoBlackwellisedParticleState(
            RaoBlackwellisedParticle(similar(x_particles), deepcopy(z_particles)),
            CUDA.zeros(T, size(x_particles, 2)),
        )
        ancestors = CuArray(1:size(x_particles, 2))

        return new{T,M,typeof(RaoBlackwellisedParticle(x_particles, z_particles))}(
            init_particles, prop_particles, ancestors
        )
    end
end

## PARTICLES ###############################################################################

"""
    ParticleState

A container for particle filters which composes the weighted sample into a distibution-like
object, with the states (or particles) distributed accoring to their log-weights.
"""
mutable struct ParticleState{PT,WT<:Real}
    particles::Vector{PT}
    log_weights::Vector{WT}
end

StatsBase.weights(state::ParticleState) = softmax(state.log_weights)

"""
    ParticleContainer

A container for information passed through each iteration of an abstract particle filter,
composed of both proposed and filtered states, as well as the ancestor indices.
"""
mutable struct ParticleContainer{T,WT}
    filtered::ParticleState{T,WT}
    proposed::ParticleState{T,WT}
    ancestors::Vector{Int}
end

function ParticleContainer(
    initial_states::Vector{T}, log_weights::Vector{WT}
) where {T,WT<:Real}
    init_particles = ParticleState(initial_states, log_weights)
    prop_particles = ParticleState(similar(initial_states), zero(log_weights))
    return ParticleContainer{T,WT}(init_particles, prop_particles, eachindex(log_weights))
end

Base.collect(state::ParticleState) = state.particles
Base.length(state::ParticleState) = length(state.particles)
Base.keys(state::ParticleState) = LinearIndices(state.particles)

# not sure if this is kosher, since it doesn't follow the convention of Base.getindex
Base.@propagate_inbounds Base.getindex(state::ParticleState, i) = state.particles[i]
# Base.@propagate_inbounds Base.getindex(state::ParticleState, i::Vector{Int}) = state.particles[i]

function reset_weights!(state::ParticleState{T,WT}) where {T,WT<:Real}
    fill!(state.log_weights, zero(WT))
    return state.log_weights
end

function update_ref!(
    proposed::ParticleState, ref_state::Union{Nothing,AbstractVector}, step::Integer=0
)
    if !isnothing(ref_state)
        proposed.particles[1] = ref_state[step]
    end
    return proposed
end

function update_ref!(
    proposed::RaoBlackwellisedParticleState,
    ref_state::Union{Nothing,AbstractVector},
    step::Integer=0,
)
    if !isnothing(ref_state)
        CUDA.@allowscalar begin
            proposed.particles.x_particles[:, 1] = ref_state[step].x_particles
            proposed.particles.z_particles.μs[:, 1] = ref_state[step].z_particles.μs
            proposed.particles.z_particles.Σs[:, :, 1] = ref_state[step].z_particles.Σs
        end
    end
    return proposed
end
