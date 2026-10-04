using BatchedKernels: BatchedKernels, TraceMatrix
using GeneralisedFilters: SqrtInformationLikelihood, SqrtBackwardInformationPredictor
using LinearAlgebra: UpperTriangular
const GF = GeneralisedFilters

GF._selected_state(x::CUDA.AnyCuVector) = BatchedCuVector(reshape(x, :, 1))
function GF._selected_component(c::BatchedStruct)
    length(c) == 1 ||
        throw(DimensionMismatch("selected-path model must return one component"))
    return c[1]
end

# Adapt small numerical primitives, retaining the shared square-root recursions.
GF._identity_plus_root(C::TraceMatrix) = BatchedKernels.qr_identity_plus(C)
GF._qr_compress_residual(B::TraceMatrix, r) = BatchedKernels.qr_compress_residual(B, r)
function GF._qr_compress_residual(B::TraceMatrix, r, C, q)
    return BatchedKernels.qr_compress_residual(B, r, C, q)
end

# The only scalar leaf here is the suffix-common log normalizer.
_single_likelihood(batch) = CUDA.@allowscalar batch[1]
const DeviceBackwardLikelihood = SqrtInformationLikelihood{
    <:CUDA.AnyCuMatrix,<:CUDA.AnyCuVector
}
const DeviceObservation = LinearGaussianObservation{
    <:CUDA.AnyCuMatrix,<:CUDA.AnyCuVector,<:CUDA.AnyCuMatrix
}

function GF.backward_initialise(
    bp::SqrtBackwardInformationPredictor, o::DeviceObservation, y
)
    return _single_likelihood(
        fuse((o, y)->GF.backward_initialise(bp, o, y), shared(o, 1), SharedCuVector(y, 1))
    )
end
function GF.backward_predict(
    bp::SqrtBackwardInformationPredictor,
    l::DeviceBackwardLikelihood,
    d::LinearGaussianDynamics,
)
    return _single_likelihood(
        fuse((l, d)->GF.backward_predict(bp, l, d), shared(l, 1), shared(d, 1))
    )
end
function GF.backward_update(
    bp::SqrtBackwardInformationPredictor,
    l::DeviceBackwardLikelihood,
    o::LinearGaussianObservation,
    y,
)
    return _single_likelihood(
        fuse(
            (l, o, y)->GF.backward_update(bp, l, o, y),
            shared(l, 1),
            shared(o, 1),
            SharedCuVector(y, 1),
        ),
    )
end
function GF.compute_marginal_predictive_likelihood(
    states::BatchedStruct{<:GaussianState}, l::DeviceBackwardLikelihood
)
    return GF.compute_marginal_predictive_likelihood.(states, shared(l, length(states)))
end

function GF.inner_dynamics(
    d::HierarchicalDynamics, t::Integer, xp::BatchedCuVector, xn::CUDA.AnyCuVector
)
    # Every candidate parent is paired with the same selected next state.
    return GF.inner_dynamics(d, t, xp, SharedCuVector(xn, length(xp)))
end
function _outer_logdensity(A, b, root, xp, xn)
    residual = UpperTriangular(root)' \ (xn - A*xp - b)
    return -(
        length(xn)*log(2π) + GF._root_logdet(UpperTriangular(root)) + sum(abs2, residual)
    )/2
end
function GF.logdensity(
    d::DeviceSamplingDynamics, ::Integer, xp::BatchedCuVector, xn::CUDA.AnyCuVector
)
    n=length(xp)
    # Outer sampling permits general covariance factors; density requires an SPD
    # covariance. Factor once per population, not once per candidate particle.
    Q = d.Q isa CovarianceFactor ? d.Q.factor*d.Q.factor' : d.Q
    root = cholesky(Hermitian(Q)).U
    return fuse(
        _outer_logdensity,
        SharedCuMatrix(d.A, n),
        SharedCuVector(d.b, n),
        SharedCuMatrix(parent(root), n),
        xp,
        SharedCuVector(xn, n),
    )
end
function _batched_ancestor_weights(states, weights, dyn, algo, t, ref)
    state=RBState(states.components.x, states.components.z)
    contribution=GF.future_conditional_density(dyn, algo, t, state, ref)
    return weights .+ contribution.data
end
function GF._ancestor_weights(
    state::ParticleDistribution{W,P,B}, dyn, algo, t, ref
) where {W,P<:Particle,B<:BatchedStruct{P}}
    return _batched_ancestor_weights(
        state.particles.components.state, GF.log_weights(state), dyn, algo, t, ref
    )
end
function GF._ancestor_weights(
    history::GF.DenseParticleContainer{T0,T,WT,V0,V}, time, dyn, algo, t, ref
) where {T0,T,WT,V0<:AbstractVector{T0},V<:BatchedStruct{T}}
    return _batched_ancestor_weights(
        history.states[time], history.weights[time], dyn, algo, t, ref
    )
end

const DeviceGaussianState = GaussianState{<:CUDA.AnyCuVector,<:CUDA.AnyCuMatrix}
# Single selected-path Kalman calculations use the same kernels as populations.
# This lets condition_inner(...), followed by the ordinary Kalman smoother, retain
# its CPU time loop and storage checks without a second smoothing implementation.
function GF._kalman_state(μ::CUDA.AnyCuVector, Σ::CUDA.AnyCuMatrix)
    return GaussianState(copy(μ), copy(Σ))
end
function GF.kalman_predict(state::DeviceGaussianState, d::LinearGaussianDynamics)
    return GF.kalman_predict.(shared(state, 1), shared(d, 1))[1]
end
function GF.kalman_update(
    state::DeviceGaussianState,
    o::LinearGaussianObservation,
    y;
    repair::GF.CovarianceRepair=NoRepair(),
)
    _check_batched_filter(KalmanFilter(; repair))
    result=GF.kalman_update.(shared(state, 1), shared(o, 1), SharedCuVector(y, 1))
    return CUDA.@allowscalar result[1]
end
function GF.rts_backward_step(
    filtered::DeviceGaussianState,
    d::LinearGaussianDynamics,
    next::GaussianState,
    predicted::Union{Nothing,GaussianState}=nothing,
)
    pred=isnothing(predicted) ? GF.kalman_predict(filtered, d) : predicted
    return GF.rts_backward_step.(
        shared(filtered, 1), shared(d, 1), shared(next, 1), shared(pred, 1)
    )[1]
end
