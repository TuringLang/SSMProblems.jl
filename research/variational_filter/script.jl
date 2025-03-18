using GeneralisedFilters, SSMProblems
using PDMats, LinearAlgebra
using Random, Distributions

using Flux, Fluxperimental
using DifferentiationInterface, Optimisers
import Mooncake

## TOY MODEL ###############################################################################

# adapted from (Naesseth, 2016)
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

## DEEP GAUSSIAN PROPOSAL ##################################################################

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

# fix for Optimisers.update! with Flux support
function Optimisers.update!(opt_state, params, grad::Mooncake.Tangent)
    return Optimisers.update!(opt_state, params, Fluxperimental._moonstrip(grad))
end

function logℓ(φ, data)
    algo = GPF(4, φ; threshold=0.8)
    _, ll = GeneralisedFilters.filter(true_model, algo, data)
    return -ll
end

num_epochs = 500
φ = DeepGaussianProposal((10,10), (16,16))
opt = Optimisers.setup(Adam(0.01), φ)
vsmc_ll = zeros(Float32, num_epochs)

backend = AutoMooncake(;config=nothing)
grad_prep = prepare_gradient(
    logℓ, backend, φ, Constant(ys)
)

@time for epoch in 1:num_epochs
    val, ∇logℓ = DifferentiationInterface.value_and_gradient(
        logℓ, grad_prep, backend, φ, Constant(ys)
    )

    Optimisers.update!(opt, φ, ∇logℓ)
    vsmc_ll[epoch] = val

    if (epoch % 25 == 0)
        println("\r$(epoch):\t -$(val)")
    else
        print("\r$(epoch):\t -$(val)")
    end
end
