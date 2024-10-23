using GeneralisedFilters
using SSMProblems
using PDMats
using LinearAlgebra
using Random: randexp
using Random
using CairoMakie
using Statistics

T = Float32
rng = MersenneTwister(1)
σx², σy² = randexp(rng, T, 2)

# initial state distribution
μ0 = zeros(T, 2)
Σ0 = PDMat(T[1 0; 0 1])

# state transition equation
A = T[1 1; 0 1]
b = T[0; 0]
Q = PDiagMat([σx²; T(1e-6)])

# observation equation
H = T[1 0]
c = T[0;]
R = [σy²;;]

# when working with PDMats, the Kalman filter doesn't play nicely without this
function Base.convert(::Type{PDMat{T,MT}}, mat::MT) where {MT<:AbstractMatrix,T<:Real}
    return PDMat(Symmetric(mat))
end

model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
_, latent, data = sample(rng, model, 100)

bf = BF(2^10; threshold=0.8)
smoother = FFBS(bf)

M = 1_000
trajectories = GeneralisedFilters.sample(rng, model, smoother, data, M)
x1 = first.(trajectories)
m = vec(mean(x1, dims=2))
stdev = vec(std(x1, dims=2))

figure = Figure()
pos = figure[1, 1]
lines(pos, m - 2 * stdev, color="black")
lines!(pos, m + 2 * stdev, color="black")
lines!(m, color="red")
lines!(pos, first.(latent), label="True latent trajectory", color="blue")
# for i in 1:M
#     lines!(pos, x1[:, i], color=:black, alpha=.01)
# end
figure