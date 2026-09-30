module GPUVolatilityExample

using GeneralisedFilters, CUDA, BatchedKernels, LinearAlgebra, Random, StaticArrays

# z_t = A z_{t-1} + b + exp(x_t/2) L ε_t; x_t is scalar log volatility.
# The same conditional model has scalar CPU and explicitly batched GPU methods.
struct VolatilityDynamics{M,V,Q}
    A::M
    b::V
    Q::Q
end
(d::VolatilityDynamics)(ctx) = LinearGaussianDynamics(d.A, d.b, exp(only(ctx.x_new)) * d.Q)

function GeneralisedFilters.inner_dynamics(
    d::VolatilityDynamics, ::Integer, xp::BatchedCuVector, xn::BatchedCuVector
)
    n = length(xn)
    k = length(d.b)
    covariance = reshape(d.Q, k, k, 1) .* reshape(exp.(xn.data), 1, 1, n)
    fields = (;
        A=SharedCuMatrix(d.A, n), b=SharedCuVector(d.b, n), Q=BatchedCuMatrix(covariance)
    )
    return BatchedStruct(LinearGaussianDynamics, fields)
end

"""Equivalent static-array CPU and device models; all fixed uploads happen here."""
function models(d=16, m=4, ::Type{T}=Float32) where {T}
    rng = Xoshiro(41)
    eye = Matrix{T}(I, d, d)
    H = randn(rng, T, m, d) / sqrt(T(d))
    # StaticArrays is the CPU baseline used in the timing comparison.
    cpu = StateSpaceModel(
        GaussianPrior(SVector{1,T}(0), SMatrix{1,1,T}(0.2)),
        LinearGaussianDynamics(
            SMatrix{1,1,T}(0.95), SVector{1,T}(0), CovarianceFactor(SMatrix{1,1,T}(0.1))
        ),
        GaussianPrior(SVector{d,T}(zeros(T, d)), SMatrix{d,d,T}(eye)),
        VolatilityDynamics(
            SMatrix{d,d,T}(T(0.9) * eye),
            SVector{d,T}(zeros(T, d)),
            SMatrix{d,d,T}(T(0.05) * eye),
        ),
        LinearGaussianObservation(
            SMatrix{m,d,T}(H),
            SVector{m,T}(zeros(T, m)),
            SMatrix{m,m,T}(T(0.2) * Matrix{T}(I, m, m)),
        ),
    )
    p, dyn, ip, id, o = cpu.prior.outer,
    cpu.dyn.outer, cpu.prior.inner, cpu.dyn.inner,
    cpu.obs.inner
    gpu = StateSpaceModel(
        GaussianPrior(CuArray(Vector(p.μ0)), CuArray(Matrix(p.Σ0))),
        LinearGaussianDynamics(
            CuArray(Matrix(dyn.A)),
            CuArray(Vector(dyn.b)),
            CovarianceFactor(CuArray(Matrix(dyn.Q.factor))),
        ),
        GaussianPrior(CuArray(Vector(ip.μ0)), CuArray(Matrix(ip.Σ0))),
        VolatilityDynamics(
            CuArray(Matrix(id.A)), CuArray(Vector(id.b)), CuArray(Matrix(id.Q))
        ),
        LinearGaussianObservation(
            CuArray(Matrix(o.H)), CuArray(Vector(o.c)), CuArray(Matrix(o.R))
        ),
    )
    return cpu, gpu
end

function observations(model, steps=20)
    rng = Xoshiro(42)
    state = simulate(rng, model.prior)
    return map(1:steps) do t
        state = simulate(rng, model.dyn, t, state)
        return simulate(rng, model.obs, t, state)
    end
end

# Temporary example-level separation until BK supports ordinary resampling draws.
# Both streams are supplied by the caller and persist across all time steps.
function filter_gpu(
    particle_rng::BatchedRNG, resampling_rng::AbstractRNG, model, algo::RBPF, ys
)
    isempty(ys) && throw(ArgumentError("filter requires nonempty observations"))
    state = initialise(particle_rng, model.prior, algo)
    total = zero(eltype(GeneralisedFilters.log_weights(state)))
    for t in eachindex(ys)
        state = GeneralisedFilters.maybe_resample(
            resampling_rng, GeneralisedFilters.resampler(algo), state
        )
        state, increment = GeneralisedFilters.move(
            particle_rng, model, algo, t, state, ys[t]
        )
        total += increment
    end
    return state, total
end

# Only the final summary is downloaded. No particle-sized host copy is needed.
function inner_mean(state)
    w = GeneralisedFilters.get_weights(state)
    if state.particles isa BatchedStruct
        return Array(state.particles.components.state.components.z.components.μ.data * w)
    end
    return sum(wi * p.state.z.μ for (wi, p) in zip(w, state.particles))
end

end
