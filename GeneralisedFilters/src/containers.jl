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

# Fields
- `particles::VT`: Vector of weighted particles
- `ll_baseline::WT`: Baseline for computing log-likelihood increment. A scalar that caches
  the unnormalized logsumexp of weights before update (for standard PF/guided filters)
  or a modified value that includes APF first-stage correction (for auxiliary PF).
"""
mutable struct ParticleDistribution{WT,PT<:Particle{<:Any,WT},VT<:AbstractVector{PT}}
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

# Helpers for StatsBase compatibility
function StatsBase.weights(state::ParticleDistribution)
    return Weights(softmax(map(p -> p.log_w, state.particles)))
end

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
function marginalise!(state::ParticleDistribution)
    # Compute logsumexp after adding observation likelihoods
    LSE_after = logsumexp(map(p -> p.log_w, state.particles))

    # Compute log-likelihood increment: works for both PF and APF cases
    ll_increment = LSE_after - state.ll_baseline

    # Normalize weights
    for p in state.particles
        p.log_w -= LSE_after
    end

    # Reset baseline for next iteration
    state.ll_baseline = 0.0

    return ll_increment
end

## GAUSSIAN STATES #########################################################################

"""
    InformationLikelihood

A container representing an unnormalized Gaussian likelihood p(y | x) in information form,
parameterized by natural parameters (λ, Ω).

The unnormalized log-likelihood is given by:
    log p(y | x) ∝ λ'x - (1/2)x'Ωx

This representation is particularly useful in backward filtering algorithms and
Rao-Blackwellised particle filtering, where it represents the predictive likelihood
p(y_{t:T} | x_t) conditioned on future observations.

# Fields
- `λ::λT`: The natural parameter vector (information vector)
- `Ω::ΩT`: The natural parameter matrix (information/precision matrix)

# See also
- [`natural_params`](@ref): Extract the natural parameters (λ, Ω)
- [`BackwardInformationPredictor`](@ref): Algorithm that uses this representation
"""
struct InformationLikelihood{λT,ΩT}
    λ::λT
    Ω::ΩT
end

"""
    natural_params(state::InformationLikelihood)

Extract the natural parameters (λ, Ω) from an InformationLikelihood.

Returns a tuple `(λ, Ω)` where λ is the information vector and Ω is the
information/precision matrix.
"""
function natural_params(state::InformationLikelihood)
    return state.λ, state.Ω
end
