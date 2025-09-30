"""Containers used for storing representations of the filtering distribution."""

## PARTICLES ###############################################################################

abstract type AbstractParticle{WT} end

# New types
# TODO (RB): could the RB particle be a regular particle with a RB state?
mutable struct Particle{ST,WT} <: AbstractParticle{WT}
    state::ST
    log_w::WT
    ancestor::Int
end

mutable struct RBParticle{XT,ZT,WT} <: AbstractParticle{WT}
    x::XT
    z::ZT
    log_w::WT
    ancestor::Int
end

mutable struct ParticleDistribution{WT,PT<:AbstractParticle{WT},VT<:AbstractVector{PT}}
    particles::VT
    prev_logsumexp::WT
end

function marginalise!(state::ParticleDistribution)
    log_marginalisation = logsumexp(map(p -> p.log_w, state.particles))
    ll_increment = (log_marginalisation - state.prev_logsumexp)
    state.prev_logsumexp = log_marginalisation
    return ll_increment
end

# Old code
mutable struct ParticleWeights{WT<:Real}
    log_weights::Vector{WT}
    prev_logsumexp::WT
end

# """
#     ParticleDistribution

# A container for particle filters which composes the weighted sample into a distibution-like
# object, with the states (or particles) distributed accoring to their log-weights.
# """
# abstract type ParticleDistribution{PT} end

# Base.collect(state::ParticleDistribution) = state.particles
# Base.length(state::ParticleDistribution) = length(state.particles)
# Base.keys(state::ParticleDistribution) = LinearIndices(state.particles)

# Base.iterate(state::ParticleDistribution, i) = iterate(state.particles, i)
# Base.iterate(state::ParticleDistribution) = iterate(state.particles)

# # not sure if this is kosher, since it doesn't follow the convention of Base.getindex
# Base.@propagate_inbounds Base.getindex(state::ParticleDistribution, i) = state.particles[i]

# mutable struct Particles{PT} <: ParticleDistribution{PT}
#     particles::Vector{PT}
#     ancestors::Vector{Int}
# end

# mutable struct WeightedParticles{PT,WT<:Real} <: ParticleDistribution{PT}
#     particles::Vector{PT}
#     ancestors::Vector{Int}
#     weights::ParticleWeights{WT}
# end

# function Particles(particles::AbstractVector)
#     N = length(particles)
#     return Particles(particles, Vector{Int}(1:N))
# end

# function WeightedParticles(particles::AbstractVector, log_weights::AbstractVector)
#     N = length(particles)
#     weights = ParticleWeights(log_weights, logsumexp(log_weights))
#     return WeightedParticles(particles, Vector{Int}(1:N), weights)
# end

# StatsBase.weights(state::Particles) = Weights(fill(1 / length(state), length(state)))
# StatsBase.weights(state::WeightedParticles) = Weights(softmax(state.weights.log_weights))

# function update_weights(state::Particles, log_weights::Vector{WT}) where {WT}
#     weights = ParticleWeights(log_weights, WT(log(length(state))))
#     return WeightedParticles(state.particles, state.ancestors, weights)
# end

# function update_weights(state::WeightedParticles, log_weights)
#     state.weights.log_weights += log_weights
#     return state
# end

# function marginalise!(state::WeightedParticles)
#     log_marginalisation = logsumexp(state.weights.log_weights)
#     ll_increment = (log_marginalisation - state.weights.prev_logsumexp)
#     state.weights.prev_logsumexp = log_marginalisation
#     return ll_increment
# end

## GAUSSIAN STATES #########################################################################

struct GaussianDistribution{PT,ΣT}
    μ::PT
    Σ::ΣT
end

function mean_cov(state::GaussianDistribution)
    return state.μ, state.Σ
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
