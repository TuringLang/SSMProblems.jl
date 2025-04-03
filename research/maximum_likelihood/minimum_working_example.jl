using LinearAlgebra
using GaussianDistributions
using Random

using DistributionsAD
using Distributions

using Enzyme

## MODEL DEFINITION ########################################################################

struct LinearGaussianProcess{
        T<:Real,
        ΦT<:AbstractMatrix{T},
        ΣT<:AbstractMatrix{T},
        μT<:AbstractVector{T}
    }
    ϕ::ΦT
    Σ::ΣT
    μ::μT
    function LinearGaussianProcess(ϕ::ΦT, Σ::ΣT, μ::μT) where {
            T<:Real,
            ΦT<:AbstractMatrix{T},
            ΣT<:AbstractMatrix{T},
            μT<:AbstractVector{T}
        }
        @assert size(ϕ,1) == size(Σ,1) == size(Σ,2) == size(μ,1)
        return new{T, ΦT, ΣT, μT}(ϕ, Σ, μ)
    end
end

# a rather simplified version of GeneralisedFilters.LinearGaussianStateSpaceModel
struct LinearGaussianModel{
        ΘT<:Real,
        TT<:LinearGaussianProcess{ΘT},
        OT<:LinearGaussianProcess{ΘT}
    }
    transition::TT
    observation::OT
end

## KALMAN FILTER ###########################################################################

# this is based on the algorithm of GeneralisedFilters.jl
function kalman_filter(
        model::LinearGaussianModel,
        init_state::Gaussian,
        observations::Vector{T}
    ) where {T<:Real}
    log_evidence = zero(T)
    filtered = init_state

    # calc_params(model.dyn)
    A = model.transition.ϕ
    Q = model.transition.Σ
    b = model.transition.μ

    # calc_params(model.obs)
    H = model.observation.ϕ
    R = model.observation.Σ
    c = model.observation.μ

    for obs in observations
        # predict step
        μ, Σ = GaussianDistributions.pair(filtered)
        proposed = Gaussian(A*μ + b, A*Σ*A' + Q)

        # update step
        μ, Σ = GaussianDistributions.pair(proposed)
        m = H*μ + c
        residual = [obs] - m

        S = Symmetric(H*Σ*H' + R)
        gain = Σ*H' / S

        filtered = Gaussian(μ + gain*residual, Σ - gain*H*Σ)
        log_evidence += logpdf(MvNormal(m, S), [obs])
    end

    return log_evidence
end

## DEMONSTRATION ###########################################################################

# model constructor
function build_model(θ::T) where {T<:Real}
    trans = LinearGaussianProcess(
        T[0.8 θ/2; -0.1 0.8],
        Diagonal(T[0.2, 1.0]),
        zeros(T, 2)
    )

    obs = LinearGaussianProcess(
        Matrix{T}(I, 1, 2),
        Diagonal(T[0.2]),
        zeros(T, 1)
    )

    return LinearGaussianModel(trans, obs)
end

# log likelihood function
function logℓ(θ::Vector{T}, data) where {T<:Real}
    model = build_model(θ[])
    init_state = Gaussian(T[1.0, 0.0], diagm(ones(T, 2)))
    return kalman_filter(model, init_state, data)
end

# refer to data globally (not preferred)
function logℓ_nodata(θ)
    return logℓ(θ, data)
end

# data generation (with unit covariance)
rng  = MersenneTwister(1234)
data = cumsum(randn(rng, 100)) .+ randn(rng, 100)

# ensure that log likelihood looks stable
logℓ([1.0], data)

## SYNTACTICAL SUGAR #######################################################################

# this has no issue behaving well
grad_test, _ = Enzyme.gradient(Enzyme.Reverse, logℓ, [1.0], Const(data))

# this error is unlegible (at least to my untrained eye)
Enzyme.hvp(logℓ_nodata, [1.0], [1.0])

## FROM SCRATCH ############################################################################

function generate_perturbations(::Type{T}, n::Int) where {T<:Real}
    perturbation_mat = Matrix{T}(I, n, n)
    return tuple(collect.(eachslice(perturbation_mat, dims=1))...)
end

generate_perturbations(n::Int) = generate_perturbations(Float64, n)
generate_perturbations(x::Vector{T}) where {T<:Real} = generate_perturbations(T, length(x))

function make_zeros(::Type{T}, n::Int) where {T<:Real}
    return tuple(collect.(zeros(T, n) for _ in 1:n)...)
end

make_zeros(n::Int) = make_zeros(Float64, n)
make_zeros(x::Vector{T}) where {T<:Real} = make_zeros(T, length(x))

function ∇logℓ(θ, args...)
    ∂θ = Enzyme.make_zero(θ)
    ∇logℓ!(θ, ∂θ, args...)
    return ∂θ
end

function ∇logℓ!(θ, ∂θ, args...)
    Enzyme.autodiff(Enzyme.Reverse, logℓ, Active, Duplicated(θ, ∂θ), args...)
    return nothing
end

# ensure I'm doing the right thing
@assert grad_test == ∇logℓ([1.0], Const(data))

# see https://enzyme.mit.edu/julia/stable/generated/autodiff/#Vector-forward-over-reverse
function hessian(θ::Vector{T}) where {T<:Real}
    # generate impulse and record second order responses
    dθ = Enzyme.make_zero(θ)
    vθ = generate_perturbations(θ)
    H  = make_zeros(θ)

    # take derivatives
    Enzyme.autodiff(
        Enzyme.Forward,
        ∇logℓ!,
        Enzyme.BatchDuplicated(θ, vθ),
        Enzyme.BatchDuplicated(dθ, H),
        Const(data),
    )

    # stack appropriately
    return vcat(H...)
end

# errors and I don't know Enzyme well enough to figure out why
hessian([1.0])
