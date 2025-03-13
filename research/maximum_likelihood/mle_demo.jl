using GeneralisedFilters
using SSMProblems
using LinearAlgebra
using Random
using DistributionsAD

## TOY MODEL ###############################################################################

# this is taken from an example in Kalman.jl
function toy_model(θ::T) where {T<:Real}
    μ0 = T[1.0, 0.0]
    Σ0 = diagm(ones(T, 2))

    A = T[0.8 θ/2; -0.1 0.8]
    Q = Diagonal(T[0.2, 1.0])
    b = zeros(T, 2)

    H = Matrix{T}(I, 1, 2)
    R = Diagonal(T[0.2])
    c = zeros(T, 1)

    return create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
end

# data generation process
rng = MersenneTwister(1234)
true_model = toy_model(1.0)
_, _, ys = sample(rng, true_model, 1000)

# evaluate and return the log evidence
function logℓ(θ, data)
    _, ll = GeneralisedFilters.filter(toy_model(θ[]), KF(), data)
    return -ll
end

# check type stability (important for use with Enzyme)
@code_warntype logℓ([1.0], ys)

## NEWTONS METHOD ##########################################################################

using DifferentiationInterface
import ForwardDiff, Zygote, Mooncake, Enzyme
using Optimisers

# Zygote will fail due to the model constructor, not because of the filtering algorithm
backends = [
    AutoZygote(), AutoForwardDiff(), AutoMooncake(;config=nothing), AutoEnzyme()
]

function gradient_descent(backend, θ_init, num_epochs=1000)
    θ = deepcopy(θ_init)
    state = Optimisers.setup(Optimisers.Descent(1/length(ys)), θ)
    grad_prep = prepare_gradient(logℓ, backend, θ, Constant(ys))

    for epoch in 1:num_epochs
        val, ∇logℓ = DifferentiationInterface.value_and_gradient(
            logℓ, grad_prep, backend, θ, Constant(ys)
        )
        Optimisers.update!(state, θ, ∇logℓ)

        (epoch % 5) == 1 && println("$(epoch-1):\t -$(val)")
        if (∇logℓ'*∇logℓ) < 1e-12
            break
        end
    end

    return θ
end

θ_init = rand(rng, 1)
for backend in backends
    println("\n",backend)
    local θ_mle
    try
        θ_mle = gradient_descent(backend, θ_init)
    catch err
        # TODO: more sophistocated exception handling
        @warn "automatic differentiation failed!" exception = (err)
    else
        # check that the solution converged to the correct value
        @assert isapprox(θ_mle, [1.0]; rtol=1e-1)
    end
end
