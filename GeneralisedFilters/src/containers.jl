"""Containers used for storing representations of the filtering distribution."""

## PARTICLES ###############################################################################

abstract type AbstractParticle{T} end

"""
    Particle

A container representing a single particle in a particle filter distribution, composed of a
weighted sampled (stored as a log weight) and its ancestor index.
"""
mutable struct Particle{ST,WT,AT<:Integer} <: AbstractParticle{ST}
    state::ST
    log_w::WT
    ancestor::AT
end

# NOTE: this is only ever used for initializing a particle filter
const UnweightedParticle{ST,AT} = Particle{ST,Nothing,AT}

Particle(state, ancestor) = Particle(state, nothing, ancestor)
Particle(particle::UnweightedParticle, ancestor) = Particle(particle.state, ancestor)
function Particle(particle::Particle{<:Any,WT}, ancestor) where {WT<:Real}
    return Particle(particle.state, zero(WT), ancestor)
end

log_weight(p::Particle{<:Any,<:Real}) = p.log_w
log_weight(::UnweightedParticle) = false

"""
    RBState

A container representing a single state with a Rao-Blackwellised component. This differs
from a `HierarchicalState` which contains a sample of the conditionally analytical state
rather than the distribution itself.

# Fields
- `x::XT`: The sampled state component
- `z::ZT`: The Rao-Blackwellised distribution component
"""
mutable struct RBState{XT,ZT}
    x::XT
    z::ZT
end

"""
    ParticleDistribution

A container for particle filters which composes a collection of weighted particles (with
their ancestories) into a distibution-like object.

# Fields
- `particles::VT`: Vector of weighted particles
- `ll_baseline::WT`: Baseline for computing log-likelihood increment. A scalar that caches
  the unnormalized logsumexp of weights before update (for standard PF/guided filters)
  or a modified value that includes APF first-stage correction (for auxiliary PF).
"""
mutable struct ParticleDistribution{WT,PT<:AbstractParticle,VT<:AbstractVector{PT}}
    particles::VT
    ll_baseline::WT
end

# Helper functions to make ParticleDistribution behave like a collection
Base.collect(state::ParticleDistribution) = state.particles
Base.length(state::ParticleDistribution) = length(state.particles)
Base.keys(state::ParticleDistribution) = LinearIndices(state.particles)
Base.iterate(state::ParticleDistribution, i) = iterate(state.particles, i)
Base.iterate(state::ParticleDistribution) = iterate(state.particles)

# Not sure if this is kosher, since it doesn't follow the convention of Base.getindex
Base.@propagate_inbounds Base.getindex(state::ParticleDistribution, i) = state.particles[i]

log_weights(state::ParticleDistribution) = map(p -> log_weight(p), state.particles)
get_weights(state::ParticleDistribution) = softmax(log_weights(state))

# Helpers for StatsBase compatibility
StatsBase.weights(state::ParticleDistribution) = StatsBase.Weights(get_weights(state))

"""
    marginalise!(state::ParticleDistribution)

Compute the log-likelihood increment and normalize particle weights. This function:
1. Computes LSE of current (post-observation) log-weights
2. Calculates ll_increment = LSE_after - ll_baseline
3. Normalizes weights by subtracting LSE_after
4. Resets ll_baseline to 0.0

The ll_baseline field handles both standard particle filter and auxiliary particle filter
cases through a single-scalar caching mechanism. For standard PF, ll_baseline equals the
LSE before adding observation weights. For APF with resampling, it includes first-stage
correction terms computed during the APF resampling step.
"""
function marginalise!(state::ParticleDistribution, particles)
    # Compute logsumexp after adding observation likelihoods
    LSE_after = logsumexp(log_weight.(particles))

    # Compute log-likelihood increment: works for both PF and APF cases
    ll_increment = LSE_after - state.ll_baseline

    # Normalize weights
    for p in particles
        p.log_w -= LSE_after
    end

    # Reset baseline for next iteration
    new_state = ParticleDistribution(particles, zero(ll_increment))
    return new_state, ll_increment
end

## GAUSSIAN STATES #########################################################################

struct GaussianDistribution{PT,ΣT}
    μ::PT
    Σ::ΣT
end

function mean_cov(state::GaussianDistribution)
    return state.μ, state.Σ
end

struct InformationDistribution{λT,ΩT}
    λ::λT
    Ω::ΩT
end

function natural_params(state::InformationDistribution)
    return state.λ, state.Ω
end

# Conversions — explicit since these may fail if the covariance/precision is not invertible
function GaussianDistribution(state::InformationDistribution)
    λ, Ω = natural_params(state)
    Σ = inv(Ω)
    μ = Σ * λ
    return GaussianDistribution(μ, Σ)
end
function InformationDistribution(state::GaussianDistribution)
    μ, Σ = mean_cov(state)
    Ω = inv(Σ)
    λ = Ω * μ
    return InformationDistribution(λ, Ω)
end
