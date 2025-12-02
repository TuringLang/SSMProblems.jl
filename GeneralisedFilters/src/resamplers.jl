using Random
using Distributions

using AcceleratedKernels: searchsortedfirst

export Multinomial, Systematic, Stratified, Metropolis, Rejection
export ESSResampler

abstract type AbstractResampler end

"""
    will_resample(resampler::AbstractResampler, state::ParticleDistribution)

Determine whether a resampler will trigger resampling given the current particle state.
For uncondition resamplers, always returns `true`. For conditional resamplers (e.g.,
`ESSResampler`), checks the resampling condition.
"""
function will_resample(::AbstractResampler, state, weights=get_weights(state))
    # Default: unconditional resamplers always resample
    return true
end

"""
    maybe_resample(
        rng::AbstractRNG,
        resampler::AbstractResampler,
        state::ParticleDistribution;
        ref_state::Union{Nothing,AbstractVector}=nothing,
    ) -> ParticleDistribution

Perform resampling if the resampler's condition is met (for conditional resamplers),
otherwise return the input state unchanged (but with ancestors set to self).
"""

function maybe_resample(
    rng::AbstractRNG,
    resampler::AbstractResampler,
    state,
    weights=get_weights(state);
    ref_state::Union{Nothing,AbstractVector}=nothing,
    auxiliary_weights=nothing,
)
    return resample(rng, resampler, state, weights; ref_state, auxiliary_weights)
end

function resample(
    rng::AbstractRNG,
    resampler::AbstractResampler,
    state,
    weights=get_weights(state);
    ref_state::Union{Nothing,AbstractVector}=nothing,
    auxiliary_weights::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    idxs = sample_ancestors(rng, resampler, weights)
    if !isnothing(ref_state)
        CUDA.@allowscalar idxs[1] = 1
    end
    return construct_new_state(state, idxs, auxiliary_weights)
end

function construct_new_state(
    state::ParticleDistribution{WT}, idxs, ::Nothing
) where {WT<:Number}
    new_particles = map(eachindex(state.particles)) do i
        particle = state.particles[idxs[i]]
        resample_ancestor(particle, idxs[i])
    end
    return ParticleDistribution(new_particles, zero(WT))
end

function resample_ancestor(particle::Particle, ancestor::Int)
    return Particle(particle, ancestor)
end

## AUXILIARY RESAMPLER #####################################################################

struct AuxiliaryResampler <: AbstractResampler
    resampler::AbstractResampler
    log_weights::AbstractVector
end

function resample(
    rng::AbstractRNG,
    auxiliary::AuxiliaryResampler,
    state;
    ref_state::Union{Nothing,AbstractVector}=nothing,
)
    weights = softmax(log_weights(state) + auxiliary.log_weights)
    auxiliary_weights = auxiliary.log_weights
    return resample(rng, auxiliary.resampler, state, weights; ref_state, auxiliary_weights)
end

function maybe_resample(
    rng::AbstractRNG, auxiliary::AuxiliaryResampler, state; ref_state=nothing
)
    weights = softmax(log_weights(state) + auxiliary.log_weights)
    auxiliary_weights = auxiliary.log_weights
    return maybe_resample(
        rng, auxiliary.resampler, state, weights; ref_state, auxiliary_weights
    )
end

function will_resample(auxiliary::AuxiliaryResampler, state, weights)
    return will_resample(auxiliary.resampler, state, weights)
end

function construct_new_state(
    state::ParticleDistribution, idxs, auxiliary_weights::AbstractVector
)
    new_particles = map(eachindex(state.particles)) do i
        particle = state.particles[idxs[i]]
        resample_ancestor(particle, idxs[i], auxiliary_weights)
    end

    # calculate the baseline log-likelihood (not a fan, but it works...)
    LSE_1 = logsumexp(auxiliary_weights + log_weights(state))
    LSE_2 = logsumexp(log_weights(state))
    LSE_3 = logsumexp(log_weight.(new_particles))
    LSE_4 = logsumexp(zero(auxiliary_weights)) # same thing as log(num_particles)

    return ParticleDistribution(new_particles, -((LSE_1 - LSE_2) + (LSE_3 - LSE_4)))
end

function resample_ancestor(
    particle::Particle, ancestor::Int, auxiliary_weights::AbstractVector
)
    return Particle(particle.state, -auxiliary_weights[ancestor], ancestor)
end

## CONDITIONAL RESAMPLING ##################################################################

abstract type AbstractConditionalResampler <: AbstractResampler end

function preserve_sample(state::ParticleDistribution)
    new_particles = map(eachindex(state.particles)) do i
        set_ancestor(state.particles[i], i)
    end
    return ParticleDistribution(new_particles, state.ll_baseline)
end

function maybe_resample(
    rng::AbstractRNG,
    cond_resampler::AbstractConditionalResampler,
    state,
    weights=get_weights(state);
    ref_state::Union{Nothing,AbstractVector}=nothing,
    auxiliary_weights::Union{Nothing,AbstractVector}=nothing,
)
    if will_resample(cond_resampler, state, weights)
        return resample(rng, cond_resampler, state, weights; ref_state, auxiliary_weights)
    else
        return preserve_sample(state)
    end
end

struct ESSResampler <: AbstractConditionalResampler
    threshold::Float64
    resampler::AbstractResampler
    function ESSResampler(threshold, resampler::AbstractResampler=Systematic())
        return new(threshold, resampler)
    end
end

function will_resample(cond_resampler::ESSResampler, state, weights=get_weights(state))
    n = length(state)
    ess = inv(sum(abs2, weights))
    return cond_resampler.threshold * n ≥ ess
end

function resample(
    rng::AbstractRNG,
    cond_resampler::ESSResampler,
    state,
    weights=get_weights(state);
    ref_state::Union{Nothing,AbstractVector}=nothing,
    auxiliary_weights::Union{Nothing,AbstractVector}=nothing,
)
    return resample(
        rng, cond_resampler.resampler, state, weights; ref_state, auxiliary_weights
    )
end

# TODO (RB): this can probably be cleaned up if we allow mutation (I'm just playing it safe
# whilst developing)
function set_ancestor(particle::Particle, ancestor::Int)
    return Particle(particle.state, log_weight(particle), ancestor)
end

## CATEGORICAL RESAMPLE ####################################################################

# this is adapted from AdvancedPS
function randcat(rng::AbstractRNG, weights::AbstractVector{WT}) where {WT<:Real}
    # pre-calculations
    @inbounds v = weights[1]
    u = rand(rng, WT)

    # initialize sampling algorithm
    n = length(weights)
    idx = 1

    while (v ≤ u) && (idx < n)
        idx += 1
        v += weights[idx]
    end

    return idx
end

## DOUBLE PRECISION STABLE ALGORITHMS ######################################################

struct Multinomial <: AbstractResampler end

function sample_ancestors(
    rng::AbstractRNG, ::Multinomial, weights::AbstractVector{WT}, n::Int64=length(weights)
) where {WT<:Real}
    return rand(rng, Distributions.Categorical(weights), n)
end

# Following Code 5 of Murray et. al (2015)
function sample_ancestors(
    rng::AbstractRNG, ::Multinomial, weights::CuVector{WT}, n::Int=length(weights)
) where {WT}
    W = cumsum(weights)
    Wn = CUDA.@allowscalar W[n]
    us = CUDA.rand(WT, n) * Wn
    as = searchsortedfirst(W, us)
    return as
end

struct Systematic <: AbstractResampler end

function sample_ancestors(
    rng::AbstractRNG, ::Systematic, weights::AbstractVector{WT}, n::Int64=length(weights)
) where {WT<:Real}
    # pre-calculations
    vs = n * cumsum(weights)
    u0 = rand(rng, WT)

    # initialize sampling algorithm
    a = Vector{Int64}(undef, n)
    idx = 1

    @inbounds for i in 1:n
        u = u0 + (i - 1)
        while vs[idx] <= u
            idx += 1
        end
        a[i] = idx
    end

    return a
end

function sample_ancestors(
    rng::AbstractRNG, ::Systematic, weights::CuVector, n::Int=length(weights)
)
    offspring = sample_offspring(rng, Systematic(), weights, n)
    return offspring_to_ancestors(offspring)
end

# Following Code 8 of Murray et. al (2015)
function sample_offspring(
    rng::AbstractRNG, ::Systematic, weights::CuVector{WT}, n::Int=length(weights)
) where {WT}
    W = cumsum(weights)
    Wn = CUDA.@allowscalar W[n]
    u0 = CUDA.@allowscalar rand(rng, WT)
    r = n * W / Wn
    offspring = min.(n, floor.(Int, r .+ u0))
    return offspring
end

struct Stratified <: AbstractResampler end

function sample_ancestors(
    rng::AbstractRNG, ::Stratified, weights::CuVector, n::Int=length(weights)
)
    offspring = sample_offspring(rng, Stratified(), weights, n)
    return offspring_to_ancestors(offspring)
end

# Following Code 7 of Murray et. al (2015)
function sample_offspring(
    rng::AbstractRNG, ::Stratified, weights::CuVector{WT}, n::Int=length(weights)
) where {WT}
    u = rand(rng, n)
    W = cumsum(weights)
    Wn = CUDA.@allowscalar W[n]
    r = n * W / Wn
    k = min.(n, floor.(Int, r .+ 1))
    offspring = min.(n, floor.(Int, r .+ u[k]))
    return offspring
end

## SINGLE PRECISION STABLE ALGORITHMS ######################################################

struct Metropolis <: AbstractResampler
    ε::Float64
    function Metropolis(ε::Float64=0.01)
        return new(ε)
    end
end

# TODO: this should be done in the log domain and also parallelized
function sample_ancestors(
    rng::AbstractRNG,
    resampler::Metropolis,
    weights::AbstractVector{WT},
    n::Int64=length(weights);
) where {WT<:Real}
    # pre-calculations
    β = mean(weights)
    B = Int64(cld(log(resampler.ε), log(1 - β)))

    # initialize the algorithm
    a = Vector{Int64}(undef, n)

    @inbounds for i in 1:n
        k = i
        for _ in 1:B
            j = rand(rng, 1:n)
            v = weights[j] / weights[k]
            if rand(rng, WT) ≤ v
                k = j
            end
        end
        a[i] = k
    end

    return a
end

struct Rejection <: AbstractResampler end

# TODO: this should be done in the log domain and also parallelized
function sample_ancestors(
    rng::AbstractRNG, ::Rejection, weights::AbstractVector{WT}, n::Int64=length(weights)
) where {WT<:Real}
    # pre-calculations
    max_weight = maximum(weights)

    # initialize the algorithm
    a = Vector{Int64}(undef, n)

    @inbounds for i in 1:n
        j = i
        u = rand(rng)
        while u > weights[j] / max_weight
            j = rand(rng, 1:n)
            u = rand(rng, WT)
        end
        a[i] = j
    end

    return a
end

## ANCESTOR-OFFSPRING CONVERSION ###########################################################

function _offspring_to_ancestors_kernel!(ancestors, offspring, N)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    @inbounds for i in index:stride:N
        start = i == 1 ? 0 : offspring[i - 1]
        finish = offspring[i]
        for j in (start + 1):finish
            ancestors[j] = i
        end
    end

    return nothing
end

function offspring_to_ancestors(offspring::CuVector{<:Integer})
    N = length(offspring)
    ancestors = similar(offspring)

    threads = 256
    blocks = ceil(Int, N / threads)

    @cuda threads = threads blocks = blocks _offspring_to_ancestors_kernel!(
        ancestors, offspring, N
    )

    return ancestors
end

function _ancestors_to_offspring_kernel!(output, ancestors, N)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    @inbounds for i in index:stride:N
        CUDA.@atomic output[ancestors[i]] += 1
    end

    return nothing
end

function ancestors_to_offspring(ancestors::CuVector{Int})
    N = length(ancestors)
    offspring = CUDA.zeros(Int, N)

    threads = 256
    blocks = ceil(Int, N / threads)

    @cuda threads = threads blocks = blocks _ancestors_to_offspring_kernel!(
        offspring, ancestors, N
    )

    return offspring
end
