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
    P = GaussianPrior{eltype(fields.μ0),eltype(fields.Σ0)}
    return BatchedStruct{P,typeof(fields)}(fields, n)
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
    D = LinearGaussianDynamics{eltype(fields.A),eltype(fields.b),eltype(fields.Q)}
    return BatchedStruct{D,typeof(fields)}(fields, n)
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
    O = LinearGaussianObservation{eltype(fields.H),eltype(fields.c),eltype(fields.R)}
    return BatchedStruct{O,typeof(fields)}(fields, n)
end

function GeneralisedFilters.simulate(
    rng::AbstractRNG, d::DeviceSamplingDynamics, ::Integer, x::BatchedCuVector{T}
) where {T}
    rng isa CUDA.RNG || throw(ArgumentError("batched Gaussian sampling requires CUDA.RNG"))
    _check_device_model_arrays(T, d.A, d.b, _sampling_storage(d.Q), x.data)
    k, n = size(x.data)
    size(d.A) == size(d.Q) == (k, k) && length(d.b) == k ||
        throw(DimensionMismatch("outer transition shapes"))
    root = _sampling_root(d.Q)
    noise = randn(rng, T, (size(root, 2), n))
    return BatchedCuVector(d.A * x.data .+ d.b .+ root * noise)
end

function GeneralisedFilters.initialise(
    rng::AbstractRNG,
    p::HierarchicalPrior{<:DeviceSamplingPrior},
    algo::RBPF{<:BootstrapFilter,<:KalmanFilter};
    ref_state=nothing,
)
    rng isa CUDA.RNG || throw(ArgumentError("batched GPU initialisation requires CUDA.RNG"))
    isnothing(ref_state) ||
        throw(ArgumentError("GPU reference trajectories are not supported"))
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
    root = _sampling_root(p.outer.Σ0)
    x = BatchedCuVector(root * randn(rng, T, (size(root, 2), n)) .+ p.outer.μ0)
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
    zfields = (; μ, Σ)
    Z = GaussianState{eltype(μ),eltype(Σ)}
    z = BatchedStruct{Z,typeof(zfields)}(zfields, n)
    statefields = (; x, z)
    S = RBState{eltype(x),eltype(z)}
    state = BatchedStruct{S,typeof(statefields)}(statefields, n)
    log_w = BatchedCuScalar(CUDA.zeros(T, n))
    ancestor = BatchedCuScalar(CUDA.zeros(Int32, n))
    particlefields = (; state, log_w, ancestor)
    P = Particle{S,T,Int32}
    particles = BatchedStruct{P,typeof(particlefields)}(particlefields, n)
    return ParticleDistribution(particles, zero(T))
end
