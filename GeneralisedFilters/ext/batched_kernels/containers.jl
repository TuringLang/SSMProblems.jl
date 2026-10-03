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

function GeneralisedFilters.construct_new_state(
    state::ParticleDistribution{W,P,B}, idxs::CuVector{<:Integer}, ::Nothing
) where {W,P<:Particle,B<:BatchedStruct{P}}
    W <: AbstractFloat ||
        throw(ArgumentError("batched particles require a numeric floating-point baseline"))
    length(idxs) == length(state) ||
        throw(DimensionMismatch("one ancestor per particle is required"))
    isempty(idxs) && throw(ArgumentError("cannot resample an empty particle batch"))
    fields = state.particles.components
    ancestors = similar(fields.ancestor.data)
    # Ancestor storage may use a narrower integer type than the resampler output.
    length(state) <= typemax(eltype(ancestors)) ||
        throw(ArgumentError("ancestor storage cannot represent every particle index"))
    ancestors .= idxs
    components = (;
        state=fields.state[idxs],
        log_w=BatchedCuScalar(zero.(fields.log_w.data)),
        ancestor=BatchedCuScalar(ancestors),
    )
    particles = BatchedStruct(Particle, components)
    return ParticleDistribution(particles, zero(W))
end

# Dense histories retain device populations and share the CPU ancestry/CSMC loops.
function GeneralisedFilters._history_states(
    state::ParticleDistribution{W,P,B}
) where {W,P<:Particle,B<:BatchedStruct{P}}
    return state.particles.components.state
end
function GeneralisedFilters._history_ancestors(
    state::ParticleDistribution{W,P,B}
) where {W,P<:Particle,B<:BatchedStruct{P}}
    return state.particles.components.ancestor.data
end

# Eager gathering copies every batched leaf, preserving explicitly shared leaves.
GeneralisedFilters._history_copy(states::BatchedStruct) = states[Base.OneTo(length(states))]

# Compact selected states: a view into the full population would keep its entire
# allocation alive when a CSMC trajectory is retained for the next sweep.
GeneralisedFilters._history_index(states::BatchedStruct, i::Integer) = states[[i]][1]

# This is an intentional single-scalar transfer during sequential ancestry tracing.
function GeneralisedFilters._history_index(xs::CUDA.AnyCuVector, i::Integer)
    CUDA.@allowscalar xs[i]
end

function GeneralisedFilters._sample_index(rng::AbstractRNG, weights::CuVector)
    index = GeneralisedFilters.sample_ancestors(
        rng, GeneralisedFilters.Multinomial(), weights, 1
    )
    return GeneralisedFilters._history_index(index, 1)
end

function GeneralisedFilters._init_tree(
    initial::ParticleDistribution, state::ParticleDistribution{W,P,B}
) where {W,P<:Particle,B<:BatchedStruct{P}}
    return GeneralisedFilters.DenseParticleContainer(initial, state)
end
