# Keep integer ancestry outside numerical fusion. Access the declared structural
# components directly rather than reconstructing individual particles on the host.
function GeneralisedFilters.log_weights(
    state::ParticleDistribution{W,P,B}
) where {W,P<:Particle,B<:BatchedStruct{P}}
    return state.particles.components.log_w.data
end

function GeneralisedFilters.get_weights(
    state::ParticleDistribution{W,P,B}
) where {W,P<:Particle,B<:BatchedStruct{P}}
    weights = GeneralisedFilters.log_weights(state)
    isempty(weights) && throw(ArgumentError("cannot normalise an empty particle batch"))
    normalizer = GeneralisedFilters._weight_logsumexp(weights)
    isfinite(normalizer) ||
        throw(ArgumentError("particle log weights have no finite normalizer"))
    return GeneralisedFilters._weight_probabilities(weights)
end

function GeneralisedFilters.preserve_sample(
    state::ParticleDistribution{W,P,B}
) where {W,P<:Particle,B<:BatchedStruct{P}}
    fields = state.particles.components
    # Match CPU ownership: preserve the state/weight leaves, replace only ancestry.
    ancestors = similar(fields.ancestor.data)
    length(state) <= typemax(eltype(ancestors)) ||
        throw(ArgumentError("ancestor storage cannot represent every particle index"))
    ancestors .= Base.OneTo(length(state))
    components = merge(fields, (; ancestor=BatchedCuScalar(ancestors)))
    particles = BatchedStruct(P, components)
    return ParticleDistribution(particles, state.ll_baseline)
end

function GeneralisedFilters.marginalise!(
    state::ParticleDistribution{W,P,B}, particles::BatchedStruct{Q}
) where {W,P<:Particle,B<:BatchedStruct{P},Q<:Particle}
    length(particles) == length(state) || throw(DimensionMismatch("particle counts differ"))
    weights = particles.components.log_w.data
    isempty(weights) && throw(ArgumentError("cannot normalise an empty particle batch"))
    logweights, ll_increment = GeneralisedFilters._normalise_logweights(
        weights, state.ll_baseline
    )
    isfinite(ll_increment) ||
        throw(ArgumentError("particle likelihood increment is not finite"))
    components = merge(particles.components, (; log_w=BatchedCuScalar(logweights)))
    normalised = BatchedStruct(Q, components)
    return ParticleDistribution(normalised, zero(ll_increment)), ll_increment
end

function GeneralisedFilters.construct_new_state(
    state::ParticleDistribution{W,P,B}, idxs, auxiliary_weights::AbstractVector
) where {W,P<:Particle,B<:BatchedStruct{P}}
    return throw(
        ArgumentError("auxiliary resampling is not supported for batched GPU particles")
    )
end

# Scalar indexing a BatchedStruct reconstructs one particle, including a device
# scalar weight/ancestor. Gather leaf arrays instead, retaining the same mapping
# across the complete nested state. No particle data is downloaded to the host.
_gather_batch(x::BatchedCuMatrix, idxs) = BatchedCuMatrix(x.data[:, :, idxs])
_gather_batch(x::BatchedCuVector, idxs) = BatchedCuVector(x.data[:, idxs])
_gather_batch(x::BatchedCuScalar, idxs) = BatchedCuScalar(x.data[idxs])
_gather_batch(x::SharedCuMatrix, idxs) = SharedCuMatrix(x.data, length(idxs))
_gather_batch(x::SharedCuVector, idxs) = SharedCuVector(x.data, length(idxs))
function _gather_batch(x::BatchedStruct{T}, idxs) where {T}
    components = map(c -> _gather_batch(c, idxs), x.components)
    # Contiguous input/output storage has identical scalar view types. Reject
    # other storage forms explicitly rather than misdeclaring composite eltypes.
    all(
        eltype(a) === eltype(b) for (a, b) in zip(values(x.components), values(components))
    ) || throw(
        ArgumentError("batched gather requires matching input/output leaf element types"),
    )
    return BatchedStruct{T,typeof(components)}(components, length(idxs))
end

function GeneralisedFilters.construct_new_state(
    state::ParticleDistribution{W,P,B}, idxs::CuVector{<:Integer}, ::Nothing
) where {W,P<:Particle,B<:BatchedStruct{P}}
    W <: AbstractFloat ||
        throw(ArgumentError("batched particles require a numeric floating-point baseline"))
    length(idxs) == length(state) ||
        throw(DimensionMismatch("one ancestor per particle is required"))
    isempty(idxs) && throw(ArgumentError("cannot resample an empty particle batch"))
    n = length(state)
    all(i -> 1 <= i <= n, idxs) || throw(BoundsError(state.particles, idxs))
    fields = state.particles.components
    ancestors = similar(fields.ancestor.data)
    # Ancestor storage may use a narrower integer type than the resampler output.
    length(state) <= typemax(eltype(ancestors)) ||
        throw(ArgumentError("ancestor storage cannot represent every particle index"))
    ancestors .= idxs
    components = (;
        state=_gather_batch(fields.state, idxs),
        log_w=BatchedCuScalar(zero.(fields.log_w.data)),
        ancestor=BatchedCuScalar(ancestors),
    )
    particles = BatchedStruct(P, components)
    return ParticleDistribution(particles, zero(W))
end
