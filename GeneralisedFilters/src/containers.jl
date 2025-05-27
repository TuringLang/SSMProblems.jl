"""Containers used for storing representations of the filtering distribution."""

## PARTICLES ###############################################################################

"""
    ParticleDistribution

A container for particle filters which composes the weighted sample into a distibution-like
object, with the states (or particles) distributed accoring to their log-weights.
"""
abstract type ParticleDistribution{PT} end

Base.collect(state::PT) where {PT<:ParticleDistribution} = state.particles
Base.length(state::PT) where {PT<:ParticleDistribution} = length(state.particles)
Base.keys(state::PT) where {PT<:ParticleDistribution} = LinearIndices(state.particles)

Base.iterate(state::ParticleDistribution, i) = Base.iterate(state.particles, i)
Base.iterate(state::ParticleDistribution) = Base.iterate(state.particles)

# not sure if this is kosher, since it doesn't follow the convention of Base.getindex
Base.@propagate_inbounds Base.getindex(state::ParticleDistribution, i) = state.particles[i]

mutable struct Particles{PT} <: ParticleDistribution{PT}
    particles::Vector{PT}
    ancestors::Vector{Int}
end

mutable struct WeightedParticles{PT,WT<:Real} <: ParticleDistribution{PT}
    particles::Vector{PT}
    ancestors::Vector{Int}
    log_weights::Vector{WT}
end

function Particles(particles::AbstractVector)
    N = length(particles)
    return Particles(particles, Vector{Int}(1:N))
end

function WeightedParticles(particles, log_weights)
    N = length(particles)
    return WeightedParticles(particles, Vector{Int}(1:N), log_weights)
end

StatsBase.weights(state::Particles) = fill(1 / length(state), length(state))
StatsBase.weights(state::WeightedParticles) = softmax(state.log_weights)

function update_weights(state::Particles, log_weights)
    return WeightedParticles(state.particles, state.ancestors, log_weights)
end

function update_weights(state::WeightedParticles, log_weights)
    state.log_weights += log_weights
    return state
end

function fast_maximum(x::AbstractArray{T}; dims)::T where {T}
    @fastmath reduce(max, x; dims, init=float(T)(-Inf))
end

function logmeanexp(x::AbstractArray{T}; dims=:)::T where {T}
    max_ = fast_maximum(x; dims)
    @fastmath max_ .+ log.(mean(exp.(x .- max_); dims))
end

## GAUSSIAN STATES #########################################################################

struct GaussianDistribution{PT,ΣT} <: ParticleDistribution{PT}
    μ::PT
    Σ::ΣT
end

function mean_cov(state::GaussianDistribution)
    return state.μ, state.Σ
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

# mutable struct RaoBlackwellisedParticleDistribution{
#     T,M<:CUDA.AbstractMemory,PT<:BatchRaoBlackwellisedParticles
# }
#     particles::PT
#     ancestors::CuVector{Int,M}
#     log_weights::CuVector{T,M}
# end
# function RaoBlackwellisedParticleDistribution(
#     particles::PT, log_weights::CuVector{T,M}
# ) where {T,M,PT}
#     N = length(log_weights)
#     return RaoBlackwellisedParticleDistribution(particles, CuVector{Int}(1:N), log_weights)
# end

# function StatsBase.weights(state::RaoBlackwellisedParticleDistribution)
#     return softmax(state.log_weights)
# end
# function Base.length(state::RaoBlackwellisedParticleDistribution)
#     return length(state.log_weights)
# end

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

# function expand(particles::CuArray{T,2,Mem}, M::Integer) where {T,Mem<:CUDA.AbstractMemory}
#     new_particles = CuArray(zeros(eltype(particles), size(particles, 1), M))
#     new_particles[:, 1:size(particles, 2)] = particles
#     return new_particles
# end

# # Method for increasing size of particle container
# function expand(p::BatchRaoBlackwellisedParticles, M::Integer)
#     new_x = expand(p.xs, M)
#     new_z = expand(p.zs, M)
#     return BatchRaoBlackwellisedParticles(new_x, new_z)
# end

# function update_ref!(
#     state::RaoBlackwellisedParticleDistribution,
#     ref_state::Union{Nothing,AbstractVector},
#     step::Integer=0,
# )
#     if !isnothing(ref_state)
#         CUDA.@allowscalar begin
#             state.particles.xs[:, 1] = ref_state[step].xs
#             state.particles.zs[1] = ref_state[step].zs
#             state.ancestors[1] = 1
#         end
#     end
#     return proposed
# end

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
