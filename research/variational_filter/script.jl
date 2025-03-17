using GeneralisedFilters
using SSMProblems
using PDMats
using LinearAlgebra
using Random
using Distributions

## LINEAR GAUSSIAN PROPOSAL ################################################################

# this is a pseudo optimal proposal kernel for linear Gaussian models
struct LinearGaussianProposal{T<:Real} <: GeneralisedFilters.AbstractProposal
    φ::Vector{T}
end

# a lot of computations done at each step
function GeneralisedFilters.distribution(
    model::AbstractStateSpaceModel,
    kernel::LinearGaussianProposal,
    step::Integer,
    state,
    observation;
    kwargs...,
)
    # get model dimensions
    dx = length(state)
    dy = length(observation)

    # see (Corenflos et al, 2021) for details
    A = GeneralisedFilters.calc_A(model.dyn, step; kwargs...)
    Γ = diagm(dx, dy, kernel.φ[(dx + 1):end])
    Σ = PDiagMat(kernel.φ[1:dx])

    return MvNormal(inv(Σ) * A * state + inv(Σ) * Γ * observation, Σ)
end

## DEEP GAUSSIAN PROPOSAL ##################################################################

using Flux, Fluxperimental

struct DeepGaussianProposal{T1,T2} <: GeneralisedFilters.AbstractProposal
    μ_net::T1
    Σ_net::T2
end

function DeepGaussianProposal(model_dims::NTuple{2, Int}, depths::NTuple{2, Int})
    input_dim = sum(model_dims)

    μnet = Chain(
        Dense(input_dim => depths[1], relu),
        Dense(depths[1] => model_dims[1])
    )

    Σnet = Chain(
        Dense(input_dim => depths[2], relu),
        Dense(depths[2] => model_dims[1], softplus)
    )

    return DeepGaussianProposal(μnet, Σnet)
end

Flux.@layer DeepGaussianProposal

function (kernel::DeepGaussianProposal)(x)
    kernel.μ_net(x), kernel.Σ_net(x)
end

function GeneralisedFilters.distribution(
    model::AbstractStateSpaceModel,
    kernel::DeepGaussianProposal,
    step::Integer,
    state,
    observation;
    kwargs...,
)
    input = cat(state, observation; dims=1)
    μ, σ = kernel(input)
    return MvNormal(μ, σ)
end

## VSMC ####################################################################################

using DifferentiationInterface
using Optimisers
using DistributionsAD
import ForwardDiff, Mooncake

function toy_model(::Type{T}, dx, dy) where {T<:Real}
    A = begin
        a = collect(1:dx)
        @. convert(T, 0.42)^(abs(a - a') + 1)
    end
    b = zeros(T, dx)
    Q = PDiagMat(ones(T, dx))

    H = diagm(dy, dx, ones(T, dy))
    c = zeros(T, dy)
    R = PDiagMat(convert(T, 0.5)*ones(T, dy))

    μ0 = zeros(T, dx)
    Σ0 = PDiagMat(ones(T, dx))

    return create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
end

rng = MersenneTwister(1234)
true_model = toy_model(Float32, 10, 10)
_, _, ys = sample(rng, true_model, 100)

function logℓ(θ, data)
    # algo = GPF(4, LinearGaussianProposal(θ); threshold=1.0)
    algo = GPF(4, θ; threshold=1.0)
    _, ll = GeneralisedFilters.filter(true_model, algo, data)
    return -ll
end

num_epochs = 500
# θ = rand(rng, Float64, 20) .+ 1.0
θ = DeepGaussianProposal((10,10), (16,16))
opt = Optimisers.setup(Adam(0.01), θ)

backend = AutoMooncake(;config=nothing)
grad_prep = prepare_gradient(
    logℓ, backend, θ, Constant(ys)
)

DifferentiationInterface.value_and_gradient(
    logℓ, AutoMooncake(;config=nothing), θ, Constant(ys)
)

@time for epoch in 1:num_epochs
    val, ∇logℓ = DifferentiationInterface.value_and_gradient(
        logℓ, grad_prep, backend, θ, Constant(ys)
    )

    Optimisers.update!(opt, θ, ∇logℓ)
    if (epoch % 25 == 0)
        println("\r$(epoch):\t -$(val)")
    else
        print("\r$(epoch):\t -$(val)")
    end
end
