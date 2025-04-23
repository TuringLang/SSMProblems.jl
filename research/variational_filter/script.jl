# # Variational Sequential Monte Carlo

#=
This example demonstrates the extensibility of GeneralisedFilters with an adaptation of VSMC
with a tunable proposal ([Naesseth et al, 2016](https://arxiv.org/pdf/1705.11140)).
=#

using GeneralisedFilters, SSMProblems
using PDMats, LinearAlgebra
using Random, Distributions

using Flux, Fluxperimental
using DifferentiationInterface, Optimisers
using Mooncake: Mooncake

using CairoMakie

# ## Model Definition

function toy_model(::Type{T}, dx, dy) where {T<:Real}
    A = begin
        a = collect(1:dx)
        @. convert(T, 0.42)^(abs(a - a') + 1)
    end
    b = zeros(T, dx)
    Q = PDiagMat(ones(T, dx))

    H = diagm(dy, dx, ones(T, dy))
    c = zeros(T, dy)
    R = PDiagMat(convert(T, 0.5) * ones(T, dy))

    μ0 = zeros(T, dx)
    Σ0 = PDiagMat(ones(T, dx))

    return create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
end

rng = MersenneTwister(1234)
true_model = toy_model(Float32, 10, 10)
_, ys = sample(rng, true_model, 100)

# ## Deep Gaussian Proposal

struct DeepGaussianProposal{T1,T2} <: GeneralisedFilters.AbstractProposal
    μ_net::T1
    Σ_net::T2
end

function DeepGaussianProposal(model_dims::NTuple{2,Int}, depths::NTuple{2,Int})
    input_dim = sum(model_dims)

    μnet = Chain(Dense(input_dim => depths[1], relu), Dense(depths[1] => model_dims[1]))

    Σnet = Chain(
        Dense(input_dim => depths[2], relu), Dense(depths[2] => model_dims[1], softplus)
    )

    return DeepGaussianProposal(μnet, Σnet)
end

Flux.@layer DeepGaussianProposal

function (kernel::DeepGaussianProposal)(x)
    return kernel.μ_net(x), kernel.Σ_net(x)
end

function SSMProblems.distribution(
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

# ## Designing the VSMC Algorithm

# fix for Optimisers.update! with Flux support
function Optimisers.update!(opt_state, params, grad::Mooncake.Tangent)
    return Optimisers.update!(opt_state, params, Fluxperimental._moonstrip(grad))
end

function logℓ(φ, data)
    algo = PF(4, φ; threshold=0.8)
    _, ll = GeneralisedFilters.filter(true_model, algo, data)
    return -ll
end

num_epochs = 500
φ = DeepGaussianProposal((10, 10), (16, 16))
opt = Optimisers.setup(Adam(0.01), φ)
vsmc_ll = zeros(Float32, num_epochs)

backend = AutoMooncake(; config=nothing)
grad_prep = prepare_gradient(logℓ, backend, φ, Constant(ys))

@time for epoch in 1:num_epochs
    ∇logℓ = DifferentiationInterface.gradient(logℓ, grad_prep, backend, φ, Constant(ys))

    Optimisers.update!(opt, φ, ∇logℓ)
    _, val = GeneralisedFilters.filter(true_model, PF(25, φ; threshold=0.8), ys)
    vsmc_ll[epoch] = val

    if (epoch % 25 == 0)
        println("\r$(epoch):\t $(val)")
    else
        print("\r$(epoch):\t $(val)")
    end
end

begin
    fig = Figure(; size=(500, 400), fontsize=16)
    ax = Axis(fig[1, 1]; limits=((0, 500), nothing), ylabel="ELBO", xlabel="Epochs")
    _, kf_ll = GeneralisedFilters.filter(true_model, KF(), ys)
    hlines!(ax, kf_ll; linewidth=3, color=:black, label="KF")
    lines!(ax, vsmc_ll; linewidth=3, color=:red, label="VSMC")
    axislegend(ax; position=:rb)
    fig
end
