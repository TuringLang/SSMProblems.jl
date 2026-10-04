module GPUVolatilityExample
using GeneralisedFilters, CUDA, BatchedKernels, Random, StaticArrays, LinearAlgebra
const GF = GeneralisedFilters

# One owner for fixed parameters. Device arrays are derived once at construction.
struct FixedParameters{H,D}
    host::H
    device::D
end
struct OuterPrior{P} <: StatePrior
    parameters::P
end
struct OuterDynamics{P} <: LatentDynamics
    parameters::P
end
struct InnerPrior{P}
    parameters::P
end
struct InnerDynamics{P,T}
    parameters::P
    logscale::T
end
struct InnerObservation{P}
    parameters::P
end

outer_prior(p) = GaussianPrior(p.μx, p.Σx)
outer_dynamics(p) = LinearGaussianDynamics(p.Ax, p.bx, CovarianceFactor(p.Lx))
inner_prior(p) = GaussianPrior(p.μz, p.Σz)
inner_observation(p) = LinearGaussianObservation(p.H, p.c, p.R)

GF.distribution(p::OuterPrior) = GF.distribution(outer_prior(p.parameters.host))
function GF.distribution(d::OuterDynamics, t::Integer, x)
    return GF.distribution(outer_dynamics(d.parameters.host), t, x)
end
function GF.simulate(rng::AbstractRNG, d::OuterDynamics, t::Integer, x::BatchedCuVector)
    return GF.simulate(rng, outer_dynamics(d.parameters.device), t, x)
end
function GF.logdensity(
    d::OuterDynamics, t::Integer, xp::BatchedCuVector, xn::CUDA.AnyCuVector
)
    return GF.logdensity(outer_dynamics(d.parameters.device), t, xp, xn)
end
(p::InnerPrior)(ctx) = inner_prior(p.parameters.host)
(o::InnerObservation)(ctx) = inner_observation(o.parameters.host)
function (d::InnerDynamics)(ctx)
    p = d.parameters.host
    return LinearGaussianDynamics(p.A, p.b, exp(d.logscale + only(ctx.x_new)) * p.Q)
end
function GF.inner_prior(p::InnerPrior, x::BatchedCuVector)
    return shared(inner_prior(p.parameters.device), length(x))
end
function GF.inner_observation(o::InnerObservation, ::Integer, x::BatchedCuVector)
    return shared(inner_observation(o.parameters.device), length(x))
end
function GF.inner_dynamics(
    d::InnerDynamics,
    ::Integer,
    xp::BatchedCuVector,
    xn::Union{BatchedCuVector,SharedCuVector},
)
    length(xp) == length(xn) || throw(DimensionMismatch("outer batch counts"))
    p, n = d.parameters.device, length(xn)
    k = length(p.b)
    covariance = if xn isa SharedCuVector
        SharedCuMatrix(p.Q .* reshape(exp.(d.logscale .+ xn.data), 1, 1), n)
    else
        BatchedCuMatrix(reshape(p.Q, k, k, 1) .* reshape(exp.(d.logscale .+ xn.data), 1, 1, n))
    end
    return BatchedStruct(
        LinearGaussianDynamics,
        (; A=SharedCuMatrix(p.A, n), b=SharedCuVector(p.b, n), Q=covariance),
    )
end

function fixed_parameters(d=16, m=4, ::Type{T}=Float32) where {T}
    rng = Xoshiro(41)
    eye = SMatrix{d,d,T}(Matrix{T}(I, d, d))
    host = (;
        μx=SVector{1,T}(0),
        Σx=SMatrix{1,1,T}(0.2),
        Ax=SMatrix{1,1,T}(0.95),
        bx=SVector{1,T}(0),
        Lx=SMatrix{1,1,T}(0.1),
        μz=zero(SVector{d,T}),
        Σz=eye,
        A=T(0.9)*eye,
        b=zero(SVector{d,T}),
        Q=T(0.05)*eye,
        H=SMatrix{m,d,T}(randn(rng, T, m, d)/sqrt(T(d))),
        c=zero(SVector{m,T}),
        R=T(0.2)*SMatrix{m,m,T}(Matrix{T}(I, m, m)),
    )
    return FixedParameters(host, map(x -> CuArray(Array(x)), host))
end

# Rebuilding at a new parameter value reuses only FIXED device storage. In
# particular, CPU AD can put a Dual in logscale without uploading any Dual arrays.
function model(p::FixedParameters, logscale=zero(eltype(p.host.b)))
    return StateSpaceModel(
        OuterPrior(p),
        OuterDynamics(p),
        InnerPrior(p),
        InnerDynamics(p, logscale),
        InnerObservation(p),
    )
end

"""One model with fixed host/device arrays and CPU/batched component methods."""
model(d=16, m=4, ::Type{T}=Float32) where {T} = model(fixed_parameters(d, m, T))

# The execution setting selects initial population storage. The rest of filtering
# uses the same model and the ordinary state-dispatched GF operations.
function GF.initialise(
    ex::GPUExecution,
    rng::AbstractRNG,
    p::HierarchicalPrior{<:OuterPrior},
    algo::RBPF{<:BootstrapFilter,<:KalmanFilter};
    ref_state=nothing,
)
    device_prior = HierarchicalPrior(outer_prior(p.outer.parameters.device), p.inner)
    return GF.initialise(ex, rng, device_prior, algo; ref_state)
end

function observations(model, steps=20)
    rng = Xoshiro(42)
    state = simulate(rng, model.prior)
    return map(1:steps) do t
        state = simulate(rng, model.dyn, t, state)
        return simulate(rng, model.obs, t, state)
    end
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
