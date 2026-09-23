using Random
using Distributions

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
    ref_idx::Integer=1,
    kwargs...,
)
    idxs = if isnothing(ref_state)
        sample_ancestors(rng, resampler, weights)
    else
        # Particle filters keep the reference trajectory in the first particle.
        conditional_sample_ancestors(rng, resampler, weights, ref_idx)
    end
    return construct_new_state(state, idxs, auxiliary_weights)
end

function construct_new_state(
    state::ParticleDistribution{WT,PT}, idxs, ::Nothing
) where {WT,PT}
    new_particles = Vector{PT}(undef, length(state.particles))
    for i in eachindex(state.particles)
        particle = state.particles[idxs[i]]
        new_particles[i] = resample_ancestor(particle, idxs[i])
    end

    return ParticleDistribution(new_particles, _zero_logweight(WT))
end

function resample_ancestor(particle::Particle, ancestor::Int)
    return Particle(particle, ancestor)
end

## CONDITIONAL SMC RESAMPLING ##############################################################

# Conditional resampling follows Finke, Johansen, Lee & Murray, "Resampling in conditional
# SMC algorithms" (arXiv:2606.25603). A scheme is characterised there by its ancestor law
# together with an index distribution λ(m, du | n) giving the slot m that particle n's
# descendant occupies. Its conditional version draws that slot, draws the scheme's auxiliary
# randomness conditioned on the reference surviving in it, and fills the remaining slots
# with the scheme's ordinary rule.
# For the returned, cyclically shifted law, integrate the native offsets and shift into
# rho_shift(a) = sum_s rho_native(shift_s(a)) / N. Each output slot then has marginal W,
# so lambda(k | n) = 1/N with no retained auxiliary variable U. Conditional draws below
# are rho_shift(a[-1] | a[1]=n). Consequently PGAS may first draw its new ancestor n and
# then call this conditional law (Algorithm 33); all remaining offspring depend on n.

"""
    supports_conditional(resampler::AbstractResampler) -> Bool

Whether `resampler` implements a conditional resampling law, and can therefore be used
within [`ConditionalSMC`](@ref). Defaults to `false`; wrappers forward the trait to the
scheme they delegate to.

This is unrelated to `AbstractConditionalResampler`, which describes resamplers that
resample only when a trigger fires (such as `ESSResampler`).
"""
supports_conditional(::AbstractResampler) = false

"""
    conditional_sample_ancestors(rng, resampler, weights, ref_idx) -> idxs

Sample ancestor indices from `resampler`'s conditional law given that particle `ref_idx`
survives resampling.

The returned indices satisfy `idxs[1] == ref_idx`: the reference occupies the first slot,
which the particle filters rely on when they propagate the reference trajectory. Only
exchangeable schemes may place the reference in a fixed slot directly (Proposition 5 of
Finke et al.); others must first be made reindexable by a random cyclical shift, so the
remaining indices come back cyclically rotated rather than in the scheme's own slot order.
That is immaterial to the filters, which treat all non-reference slots alike.

Drawing unconditionally and overwriting one index afterwards is *not* a conditional law for
any scheme whose ancestors are dependent or not identically distributed. It biases the CSMC
kernel and worsens its autocorrelation (Section 5.3 of the reference), which is why this is
a separate method rather than a post-hoc fixup.
"""
function conditional_sample_ancestors(
    ::AbstractRNG, resampler::AbstractResampler, weights, ref_idx::Integer
)
    return throw(
        ArgumentError(
            "$(nameof(typeof(resampler))) does not implement conditional resampling and " *
            "cannot be used for conditional SMC",
        ),
    )
end

"""
    _reference_offset(rng, vs, ref_idx, n) -> (u, K)

Draw the reference particle's slot `K` and the stratum offset `u` belonging to it, from the
index distribution shared by stratified and systematic resampling.

`vs` is the scaled cumulative weight vector `n * F`, in which slot `m` covers `(m - 1, m]`
and particle `n` covers `(vs[n - 1], vs[n]]`. Drawing `V` uniformly on the reference
particle's own interval and splitting it into integer and fractional parts therefore
realises `K ~ Cat((vol(I^m(ref_idx)) / (n * W[ref_idx]))_m)` and `u ~ Unif(I^K(ref_idx))`
jointly, without forming those intervals explicitly.
"""
function _reference_offset(
    rng::AbstractRNG, vs::AbstractVector{WT}, ref_idx::Integer, n::Integer
) where {WT<:Real}
    return _reference_offset(rand(rng, WT), vs, ref_idx, n)
end

# Separate the uniform draw from the law so device RNGs can supply a host scalar
# without requiring Random's scalar sampling API.
function _reference_offset(
    u::Real, vs::AbstractVector{WT}, ref_idx::Integer, n::Integer
) where {WT<:Real}
    lower = ref_idx == 1 ? zero(WT) : vs[ref_idx - 1]
    v = lower + (vs[ref_idx] - lower) * u
    # `min` guards against weights that sum to marginally more than one.
    K = min(floor(Int, v) + 1, n)
    return v - (K - 1), K
end

## AUXILIARY RESAMPLER #####################################################################

"""
    AuxiliaryResampler

A resampling scheme for multistage particle resampling with auxiliary weights
"""
struct AuxiliaryResampler <: AbstractResampler
    resampler::AbstractResampler
    log_weights::AbstractVector
end

function resample(
    rng::AbstractRNG,
    auxiliary::AuxiliaryResampler,
    state;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    ref_idx::Integer=1,
)
    weights = softmax(add_logweight.(log_weights(state), auxiliary.log_weights))
    auxiliary_weights = auxiliary.log_weights
    return resample(
        rng, auxiliary.resampler, state, weights; ref_state, ref_idx, auxiliary_weights
    )
end

function maybe_resample(
    rng::AbstractRNG, auxiliary::AuxiliaryResampler, state; ref_state=nothing
)
    weights = softmax(add_logweight.(log_weights(state), auxiliary.log_weights))
    auxiliary_weights = auxiliary.log_weights
    return maybe_resample(
        rng, auxiliary.resampler, state, weights; ref_state, auxiliary_weights
    )
end

function will_resample(auxiliary::AuxiliaryResampler, state)
    weights = softmax(add_logweight.(log_weights(state), auxiliary.log_weights))
    return will_resample(auxiliary.resampler, state, weights)
end
function will_resample(auxiliary::AuxiliaryResampler, state, weights)
    return will_resample(auxiliary.resampler, state, weights)
end

function supports_conditional(auxiliary::AuxiliaryResampler)
    return supports_conditional(auxiliary.resampler)
end

function construct_new_state(
    state::ParticleDistribution, idxs, auxiliary_weights::AbstractVector
)
    new_particles = map(eachindex(state.particles)) do i
        particle = state.particles[idxs[i]]
        return resample_ancestor(particle, idxs[i], auxiliary_weights)
    end

    # Preserve the APF first-stage correction, resolving count normalisers only
    # against the numeric lookahead contributions.
    LSE_1 = _weight_logsumexp(add_logweight.(auxiliary_weights, log_weights(state)))
    LSE_2 = _weight_logsumexp(log_weights(state))
    LSE_3 = _weight_logsumexp(log_weight.(new_particles))
    LSE_4 = TypelessBaseline(length(auxiliary_weights))

    baseline =
        -add_logweight(_subtract_baseline(LSE_1, LSE_2), _subtract_baseline(LSE_3, LSE_4))
    return ParticleDistribution(new_particles, baseline)
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
        return set_ancestor(state.particles[i], i)
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

struct ESSResampler{RS<:AbstractResampler} <: AbstractConditionalResampler
    threshold::Float64
    resampler::RS
    function ESSResampler(threshold, resampler::AbstractResampler=Systematic())
        return new{typeof(resampler)}(threshold, resampler)
    end
end

function supports_conditional(cond_resampler::ESSResampler)
    return supports_conditional(cond_resampler.resampler)
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
    ref_idx::Integer=1,
)
    return resample(
        rng, cond_resampler.resampler, state, weights; ref_state, ref_idx, auxiliary_weights
    )
end

# TODO (RB): this can probably be cleaned up if we allow mutation (I'm just playing it safe
# whilst developing)
function set_ancestor(particle::Particle, ancestor::Int)
    return Particle(particle.state, log_weight(particle), ancestor)
end

## DOUBLE PRECISION STABLE ALGORITHMS ######################################################

struct Multinomial <: AbstractResampler end

function sample_ancestors(
    rng::AbstractRNG, ::Multinomial, weights::AbstractVector{WT}, n::Int64=length(weights)
) where {WT<:Real}
    return rand(rng, Distributions.Categorical(weights), n)
end

supports_conditional(::Multinomial) = true

# Multinomial resampling is the one elementary scheme that is exchangeable, so the reference
# slot can be fixed rather than drawn from the index distribution, and the remaining slots
# are drawn by the ordinary rule.
function conditional_sample_ancestors(
    rng::AbstractRNG, ::Multinomial, weights::AbstractVector{WT}, ref_idx::Integer
) where {WT<:Real}
    n = length(weights)
    a = Vector{Int64}(undef, n)
    a[1] = ref_idx
    if n > 1
        a[2:n] = rand(rng, Distributions.Categorical(weights), n - 1)
    end
    return a
end

struct Systematic <: AbstractResampler end

function sample_ancestors(
    rng::AbstractRNG, ::Systematic, weights::AbstractVector{WT}, n::Int64=length(weights)
) where {WT<:Real}
    # pre-calculations
    vs = cumsum(weights)
    vs *= n

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

supports_conditional(::Systematic) = true

# Systematic resampling is not exchangeable, so the reference slot `K` is drawn from the
# index distribution and the indices are then cyclically shifted by `K - 1` to bring it to
# the front. Only the shared offset differs from the unconditional scheme.
function conditional_sample_ancestors(
    rng::AbstractRNG, ::Systematic, weights::AbstractVector{WT}, ref_idx::Integer
) where {WT<:Real}
    n = length(weights)

    vs = cumsum(weights)
    vs *= n

    u0, K = _reference_offset(rng, vs, ref_idx, n)

    a = Vector{Int64}(undef, n)
    idx = 1

    @inbounds for i in 1:n
        u = u0 + (i - 1)
        while vs[idx] <= u
            idx += 1
        end
        a[mod1(i - K + 1, n)] = idx
    end

    return a
end

struct Stratified <: AbstractResampler end

function sample_ancestors(
    rng::AbstractRNG, ::Stratified, weights::AbstractVector{WT}, n::Int64=length(weights)
) where {WT<:Real}
    # pre-calculations
    vs = cumsum(weights)
    vs *= n

    # initialize sampling algorithm
    a = Vector{Int64}(undef, n)
    idx = 1

    @inbounds for i in 1:n
        u = rand(rng, WT) + (i - 1)
        while vs[idx] <= u
            idx += 1
        end
        a[i] = idx
    end

    return a
end

supports_conditional(::Stratified) = true

# As for systematic resampling, the reference slot is drawn and then shifted to the front;
# here every other slot keeps its own independent offset.
function conditional_sample_ancestors(
    rng::AbstractRNG, ::Stratified, weights::AbstractVector{WT}, ref_idx::Integer
) where {WT<:Real}
    n = length(weights)

    vs = cumsum(weights)
    vs *= n

    u_ref, K = _reference_offset(rng, vs, ref_idx, n)

    a = Vector{Int64}(undef, n)
    idx = 1

    @inbounds for i in 1:n
        u = (i == K ? u_ref : rand(rng, WT)) + (i - 1)
        while vs[idx] <= u
            idx += 1
        end
        a[mod1(i - K + 1, n)] = idx
    end

    return a
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
