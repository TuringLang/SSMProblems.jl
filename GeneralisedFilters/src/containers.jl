"""Containers used for storing representations of the filtering distribution."""

## PARTICLES ###############################################################################

"""
    ParticleDistribution

A container for particle filters which composes the weighted sample into a distibution-like
object, with the states (or particles) distributed accoring to their log-weights.
"""
abstract type ParticleDistribution{PT} end

Base.collect(state::ParticleDistribution) = state.particles
Base.length(state::ParticleDistribution) = length(state.particles)
Base.keys(state::ParticleDistribution) = LinearIndices(state.particles)

Base.iterate(state::ParticleDistribution, i) = iterate(state.particles, i)
Base.iterate(state::ParticleDistribution) = iterate(state.particles)

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

struct GaussianDistribution{PT,ΣT}
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
