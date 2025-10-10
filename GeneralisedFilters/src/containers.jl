"""Containers used for storing representations of the filtering distribution."""

## PARTICLES ###############################################################################

"""
    Particle

A container representing a single particle in a particle filter distribution, composed of a
weighted sampled (stored as a log weight) and its ancestor index.
"""
mutable struct Particle{ST,WT,AT<:Integer}
    state::ST
    log_w::WT
    ancestor::AT
end

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

const RBParticle{XT,ZT,WT} = Particle{RBState{XT,ZT},WT}

"""
    ParticleDistribution

A container for particle filters which composes a collection of weighted particles (with
their ancestories) into a distibution-like object.
"""
mutable struct ParticleDistribution{WT,PT<:Particle{<:Any,WT},VT<:AbstractVector{PT}}
    particles::VT
    prev_logsumexp::WT
end

# Helper functions to make ParticleDistribution behave like a collection
Base.collect(state::ParticleDistribution) = state.particles
Base.length(state::ParticleDistribution) = length(state.particles)
Base.keys(state::ParticleDistribution) = LinearIndices(state.particles)
Base.iterate(state::ParticleDistribution, i) = iterate(state.particles, i)
Base.iterate(state::ParticleDistribution) = iterate(state.particles)

# Not sure if this is kosher, since it doesn't follow the convention of Base.getindex
Base.@propagate_inbounds Base.getindex(state::ParticleDistribution, i) = state.particles[i]

# Helpers for StatsBase compatibility
function StatsBase.weights(state::ParticleDistribution)
    return Weights(softmax(map(p -> p.log_w, state.particles)))
end

"""
    marginalise!(state::ParticleDistribution)

Compute the current log marginal likelihood and store for the next iteration. In the
process, return the increment in log marginal likelihood. Note that `state.prev_logsumexp`
also gets reset when resampling occurs.
"""
function marginalise!(state::ParticleDistribution)
    log_marginalisation = logsumexp(map(p -> p.log_w, state.particles))
    ll_increment = (log_marginalisation - state.prev_logsumexp)
    state.prev_logsumexp = log_marginalisation
    return ll_increment
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
