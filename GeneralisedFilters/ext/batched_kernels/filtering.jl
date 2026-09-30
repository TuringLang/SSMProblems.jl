using GeneralisedFilters: HierarchicalDynamics, HierarchicalObservation, HierarchicalSSM
using Random: AbstractRNG

# Packing is a structural operation: no numerical kernel or integer conversion.
function _rb_particles(state::RBState, log_w, ancestor)
    states = BatchedStruct(RBState, (; x=state.x, z=state.z))
    return BatchedStruct(Particle, (; state=states, log_w, ancestor))
end

GeneralisedFilters.add_logweight(w::BatchedCuScalar, ::GeneralisedFilters.TypelessZero) = w
function GeneralisedFilters.add_logweight(w::BatchedCuScalar, increment::BatchedCuScalar)
    return BatchedCuScalar(w.data .+ increment.data)
end

function _check_batched_filter(algo::KalmanFilter, ref_state=nothing)
    isnothing(ref_state) ||
        throw(ArgumentError("GPU reference trajectories are not supported"))
    algo.repair isa NoRepair ||
        throw(ArgumentError("batched Kalman filtering requires NoRepair"))
    return nothing
end

# Validate before the generic step attempts conditional resampling. Keep its
# resampling, move and weight-type checks as the single implementation.
function GeneralisedFilters.step(
    rng::AbstractRNG,
    model::HierarchicalSSM,
    algo::RBPF{<:BootstrapFilter,<:KalmanFilter},
    t::Integer,
    state::ParticleDistribution{W,P,B},
    y;
    ref_state::Union{Nothing,AbstractVector}=nothing,
) where {W,P<:Particle{<:RBState},B<:BatchedStruct{P}}
    _check_batched_filter(algo.af, ref_state)
    return invoke(
        GeneralisedFilters.step,
        Tuple{
            AbstractRNG,
            GeneralisedFilters.AbstractStateSpaceModel,
            GeneralisedFilters.AbstractParticleFilter,
            Integer,
            Any,
            Any,
        },
        rng,
        model,
        algo,
        t,
        state,
        y;
        ref_state,
    )
end

# Bulk execution changes traversal and packing, not the RBPF recipe. The plain
# RBState here contains batched fields and is never a single sampled particle.
function GeneralisedFilters._predict_particles(
    rng::AbstractRNG,
    dyn::HierarchicalDynamics,
    algo::RBPF{<:BootstrapFilter,<:KalmanFilter},
    t::Integer,
    particles::BatchedStruct{P},
    observation,
    ref_state,
) where {P<:Particle{<:RBState}}
    _check_batched_filter(algo.af, ref_state)
    fields = particles.components
    state = RBState(fields.state.components.x, fields.state.components.z)
    predicted, increment = GeneralisedFilters._predict_rb_state(
        rng, dyn, algo, t, state, observation, ref_state
    )
    weights = GeneralisedFilters.add_logweight(fields.log_w, increment)
    return _rb_particles(predicted, weights, fields.ancestor)
end

function GeneralisedFilters._update_particles(
    obs::HierarchicalObservation,
    algo::RBPF{<:BootstrapFilter,<:KalmanFilter},
    t::Integer,
    particles::BatchedStruct{P},
    observation,
) where {P<:Particle{<:RBState}}
    _check_batched_filter(algo.af)
    fields = particles.components
    state = RBState(fields.state.components.x, fields.state.components.z)
    filtered, increment = GeneralisedFilters._update_rb_state(
        obs, algo, t, state, observation
    )
    weights = GeneralisedFilters.add_logweight(fields.log_w, increment)
    return _rb_particles(filtered, weights, fields.ancestor)
end

# Adapt batched model/belief containers to the shared numerical API.
# BatchedKernels owns broadcast execution and kernel fusion.
function GeneralisedFilters.predict(
    ::AbstractRNG,
    dyn::BatchedStruct{<:LinearGaussianDynamics},
    algo::KalmanFilter,
    ::Integer,
    state::BatchedStruct{<:GaussianState},
    observation;
    ref_state=nothing,
)
    _check_batched_filter(algo, ref_state)
    return GeneralisedFilters.kalman_predict.(state, dyn)
end

function GeneralisedFilters.update(
    obs::BatchedStruct{<:LinearGaussianObservation},
    algo::KalmanFilter,
    ::Integer,
    state::BatchedStruct{<:GaussianState},
    y,
)
    _check_batched_filter(algo)
    T = eltype(state.components.μ.data)
    y isa CuVector{T} || throw(
        ArgumentError(
            "batched Kalman observations must be CuVectors matching the belief precision",
        ),
    )
    result = GeneralisedFilters.kalman_update.(state, obs, SharedCuVector(y, length(state)))
    filtered, increments = result.components
    # Reject non-finite likelihoods before normalisation. CUDA kernel failures
    # (including a negative Cholesky pivot) propagate before this check.
    all(isfinite, increments.data) || throw(
        ArgumentError(
            "batched Kalman update produced a non-finite likelihood; check innovation covariances",
        ),
    )
    return filtered, increments
end
