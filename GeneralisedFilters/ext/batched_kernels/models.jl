using GeneralisedFilters:
    GaussianPrior,
    LinearGaussianDynamics,
    LinearGaussianObservation,
    HierarchicalPrior,
    RBPF,
    BootstrapFilter,
    KalmanFilter,
    NoRepair,
    GaussianState,
    RBState,
    CovarianceFactor
using Random: AbstractRNG, randn
using LinearAlgebra: cholesky, Hermitian

# Dispatch aliases only: these are existing model types, not device wrappers.
# Sampling accepts covariance factors; inner Kalman atoms currently use matrices.
const DeviceCovariance = Union{CuMatrix,CovarianceFactor{<:Any,<:CuMatrix}}
const DeviceSamplingPrior = GaussianPrior{<:CuVector,<:DeviceCovariance}
const DeviceSamplingDynamics = LinearGaussianDynamics{
    <:CuMatrix,<:CuVector,<:DeviceCovariance
}

# Match CPU Gaussian sampling's upper-triangle convention. Multiply by the
# adjoint directly: extracting .L would materialise via a generic scalar copy.
_sampling_root(Q::CuMatrix) = cholesky(Hermitian(Q)).U'
_sampling_root(Q::CovarianceFactor{<:Any,<:CuMatrix}) = Q.factor
_sampling_storage(Q::CuMatrix) = Q
_sampling_storage(Q::CovarianceFactor{<:Any,<:CuMatrix}) = Q.factor

# Model constructors declare device residency; do not upload fixed parameters in
# per-step resolution. Shared wrappers express sharing without comparing values.
function _check_device_model_arrays(::Type{T}, arrays...) where {T}
    T in (Float32, Float64) ||
        throw(ArgumentError("batched Gaussian models require Float32 or Float64"))
    all(a -> a isa CuArray && eltype(a) === T, arrays) || throw(
        ArgumentError(
            "batched Gaussian model arrays must be device arrays of one floating-point type",
        ),
    )
    return nothing
end

function GeneralisedFilters.inner_prior(
    p::GaussianPrior{<:CuVector,<:CuMatrix}, x::BatchedCuVector{T}
) where {T}
    _check_device_model_arrays(T, p.μ0, p.Σ0, x.data)
    d = length(p.μ0)
    size(p.Σ0) == (d, d) || throw(DimensionMismatch("inner prior covariance shape"))
    n = length(x)
    fields = (; μ0=SharedCuVector(p.μ0, n), Σ0=SharedCuMatrix(p.Σ0, n))
    return BatchedStruct(GaussianPrior, fields)
end

function GeneralisedFilters.inner_dynamics(
    d::LinearGaussianDynamics{<:CuMatrix,<:CuVector,<:CuMatrix},
    ::Integer,
    xp::BatchedCuVector{T},
    xn::BatchedCuVector,
) where {T}
    length(xp) == length(xn) || throw(DimensionMismatch("outer state batch counts differ"))
    _check_device_model_arrays(T, d.A, d.b, d.Q, xp.data, xn.data)
    k = length(d.b)
    size(d.A) == size(d.Q) == (k, k) || throw(DimensionMismatch("inner dynamics shapes"))
    n = length(xp)
    fields = (;
        A=SharedCuMatrix(d.A, n), b=SharedCuVector(d.b, n), Q=SharedCuMatrix(d.Q, n)
    )
    return BatchedStruct(LinearGaussianDynamics, fields)
end

function GeneralisedFilters.inner_observation(
    o::LinearGaussianObservation{<:CuMatrix,<:CuVector,<:CuMatrix},
    ::Integer,
    x::BatchedCuVector{T},
) where {T}
    _check_device_model_arrays(T, o.H, o.c, o.R, x.data)
    k = length(o.c)
    size(o.H, 1) == k && size(o.R) == (k, k) ||
        throw(DimensionMismatch("inner observation shapes"))
    n = length(x)
    fields = (;
        H=SharedCuMatrix(o.H, n), c=SharedCuVector(o.c, n), R=SharedCuMatrix(o.R, n)
    )
    return BatchedStruct(LinearGaussianObservation, fields)
end

# Factor shared covariances once on the host-controlled path, then fuse the
# per-particle draw and affine transformation.
_gaussian_draw(rng, μ, root) = μ + root * randn(rng, eltype(root), size(root, 2))
_gaussian_transition(rng, A, b, root, x) = A * x + _gaussian_draw(rng, b, root)

function _draw_outer(rng::CUDA.RNG, μ, root, n)
    return BatchedCuVector(root * randn(rng, eltype(μ), (size(root, 2), n)) .+ μ)
end
function _draw_outer(rng::BatchedRNG, μ, root, n)
    return fuse(_gaussian_draw, rng, SharedCuVector(μ, n), SharedCuMatrix(root, n))
end
function _transition_outer(rng::CUDA.RNG, A, b, root, x)
    noise = randn(rng, eltype(x.data), (size(root, 2), length(x)))
    return BatchedCuVector(A * x.data .+ b .+ root * noise)
end
function _transition_outer(rng::BatchedRNG, A, b, root, x)
    n = length(x)
    return fuse(
        _gaussian_transition,
        rng,
        SharedCuMatrix(A, n),
        SharedCuVector(b, n),
        SharedCuMatrix(root, n),
        x,
    )
end

function GeneralisedFilters.simulate(
    rng::AbstractRNG, d::DeviceSamplingDynamics, ::Integer, x::BatchedCuVector{T}
) where {T}
    rng = GeneralisedFilters.gpu_rng(rng)
    rng isa Union{CUDA.RNG,BatchedRNG} ||
        throw(ArgumentError("batched Gaussian sampling requires CUDA.RNG or BatchedRNG"))
    _check_device_model_arrays(T, d.A, d.b, _sampling_storage(d.Q), x.data)
    k, n = size(x.data)
    size(d.A) == size(d.Q) == (k, k) && length(d.b) == k ||
        throw(DimensionMismatch("outer transition shapes"))
    root = _sampling_root(d.Q)
    return _transition_outer(rng, d.A, d.b, root, x)
end

# GF retains reference residency/precision validation; BK owns member assignment.
_pin_reference!(x::BatchedCuVector, ::Nothing) = x
function _pin_reference!(x::BatchedCuVector{T}, ref) where {T}
    ref isa CUDA.AnyCuVector{T} || throw(
        ArgumentError(
            "GPU reference states must be device vectors matching the particle precision",
        ),
    )
    x[1] = ref
    return x
end

function GeneralisedFilters.initialise(
    rng::AbstractRNG,
    p::HierarchicalPrior{<:DeviceSamplingPrior},
    algo::RBPF{<:BootstrapFilter,<:KalmanFilter};
    ref_state=nothing,
)
    rng = GeneralisedFilters.gpu_rng(rng)
    rng isa Union{CUDA.RNG,BatchedRNG} ||
        throw(ArgumentError("batched GPU initialisation requires CUDA.RNG or BatchedRNG"))
    algo.af.repair isa NoRepair ||
        throw(ArgumentError("batched Kalman filtering requires NoRepair"))
    T = eltype(p.outer.μ0)
    _check_device_model_arrays(T, p.outer.μ0, _sampling_storage(p.outer.Σ0))
    k = length(p.outer.μ0)
    size(p.outer.Σ0) == (k, k) || throw(DimensionMismatch("outer prior covariance shape"))
    all(isfinite, p.outer.μ0) && all(isfinite, _sampling_storage(p.outer.Σ0)) ||
        throw(ArgumentError("outer prior parameters must be finite"))
    n = GeneralisedFilters.num_particles(algo)
    n <= typemax(Int32) ||
        throw(ArgumentError("particle count exceeds Int32 ancestor storage"))
    ref = GeneralisedFilters._reference_state(ref_state, 0)
    root = _sampling_root(p.outer.Σ0)
    x = _pin_reference!(_draw_outer(rng, p.outer.μ0, root, n), ref)
    ip = GeneralisedFilters.inner_prior(p, x)
    ip isa BatchedStruct{<:GaussianPrior} || throw(
        ArgumentError("GPU inner_prior must return a BatchedStruct of GaussianPrior atoms"),
    )
    length(ip) == n || throw(DimensionMismatch("inner prior batch count"))
    fields = ip.components
    fields.μ0 isa Union{SharedCuVector,BatchedCuVector} &&
    fields.Σ0 isa Union{SharedCuMatrix,BatchedCuMatrix} || throw(
        ArgumentError("inner prior requires shared or batched device vector/matrix leaves"),
    )
    length(fields.μ0) == length(fields.Σ0) == n ||
        throw(DimensionMismatch("inner prior leaf batch counts"))
    _check_device_model_arrays(T, fields.μ0.data, fields.Σ0.data)
    d = size(fields.μ0.data, 1)
    size(fields.Σ0.data)[1:2] == (d, d) || throw(DimensionMismatch("inner prior shapes"))
    all(isfinite, fields.μ0.data) && all(isfinite, fields.Σ0.data) ||
        throw(ArgumentError("inner prior parameters must be finite"))
    # Initial beliefs are values, not a numerical operation needing fusion.
    # Copy model-owned leaves so the returned state owns its mutable storage.
    μ = BatchedCuVector(
        fields.μ0 isa SharedCuVector ? repeat(fields.μ0.data, 1, n) : copy(fields.μ0.data)
    )
    Σ = BatchedCuMatrix(
        if fields.Σ0 isa SharedCuMatrix
            repeat(fields.Σ0.data, 1, 1, n)
        else
            copy(fields.Σ0.data)
        end,
    )
    z = BatchedStruct(GaussianState, (; μ, Σ))
    state = BatchedStruct(RBState, (; x, z))
    log_w = BatchedCuScalar(CUDA.zeros(T, n))
    ancestor = BatchedCuScalar(CUDA.zeros(Int32, n))
    particles = BatchedStruct(Particle, (; state, log_w, ancestor))
    return ParticleDistribution(particles, zero(T))
end
