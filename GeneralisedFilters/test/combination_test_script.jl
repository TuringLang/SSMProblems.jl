using Distributions
using GeneralisedFilters
using LinearAlgebra
using LogExpFunctions
using SSMProblems
using StableRNGs
using StatsBase
using Test

println()
println("########################")
println("#### STARTING TESTS ####")
println("########################")
println()

rng = StableRNG(1234)

model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, 1, 1)
_, _, ys = sample(rng, model, 3)

bf = BF(10^6; threshold=0.8)
bf_state, llbf = GeneralisedFilters.filter(rng, model, bf, ys)
kf_state, llkf = GeneralisedFilters.filter(rng, model, KF(), ys)

xs = getfield.(bf_state.particles, :state)
log_ws = getfield.(bf_state.particles, :log_w)
ws = softmax(log_ws)

# Compare log-likelihood and states
println("BF State: ", @test first(kf_state.μ) ≈ sum(first.(xs) .* ws) rtol = 1e-3)
println("BF LL: ", @test llkf ≈ llbf atol = 1e-3)

struct OptimalProposal <: AbstractProposal
    dyn::LinearGaussianLatentDynamics
    obs::LinearGaussianObservationProcess
    dummy::Bool  # if using dummy hierarchical model
end
function SSMProblems.distribution(prop::OptimalProposal, step::Integer, x, y; kwargs...)
    A, b, Q = GeneralisedFilters.calc_params(prop.dyn, step; kwargs...)
    H, c, R = GeneralisedFilters.calc_params(prop.obs, step; kwargs...)
    Σ = inv(inv(Q) + H' * inv(R) * H)
    μ = Σ * (inv(Q) * (A * x + b) + H' * inv(R) * (y - c))
    if prop.dummy
        μ = μ[[1]]
        Σ = Σ[[1], [1]]
    end
    return MvNormal(μ, Σ)
end
# Propose from observation distribution
# proposal = PeturbationProposal(only(model.obs.R))
proposal = OptimalProposal(model.dyn, model.obs, false)
gf = ParticleFilter(10^6, proposal; threshold=1.0)

gf_state, llgf = GeneralisedFilters.filter(rng, model, gf, ys)
xs = getfield.(gf_state.particles, :state)
log_ws = getfield.(gf_state.particles, :log_w)
ws = softmax(log_ws)

# Fairly sure this is correct but would be good to confirm (needs to be faster — SArrays)
println("GF State: ", @test first(kf_state.μ) ≈ sum(first.(xs) .* ws) rtol = 1e-3)
println("GF LL: ", @test llkf ≈ llgf atol = 1e-3)

##############################
#### RAO-BLACKWELLISATION ####
##############################

full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
    rng, 1, 1, 1; static_arrays=true
)
_, _, ys = sample(rng, hier_model, 3)

rbbf = RBPF(bf, KalmanFilter())

rbbf_state, llrbbf = GeneralisedFilters.filter(rng, hier_model, rbbf, ys)
xs = getfield.(rbbf_state.particles, :x)
zs = getfield.(rbbf_state.particles, :z)
log_ws = getfield.(rbbf_state.particles, :log_w)
ws = softmax(log_ws)

kf_state, llkf = GeneralisedFilters.filter(rng, full_model, KF(), ys)

println("RBBF Outer: ", @test first(kf_state.μ) ≈ sum(only.(xs) .* ws) rtol = 1e-3)
println(
    "RBBF Inner: ", @test last(kf_state.μ) ≈ sum(only.(getfield.(zs, :μ)) .* ws) rtol = 1e-3
)
println("RBBF LL: ", @test llkf ≈ llrbbf atol = 1e-3)

proposal = OptimalProposal(model.dyn, model.obs, true)
gf = ParticleFilter(10^6, proposal; threshold=1.0)
rbgf = RBPF(gf, KalmanFilter())
rbgf_state, llrbgf = GeneralisedFilters.filter(rng, hier_model, rbgf, ys)
xs = getfield.(rbgf_state.particles, :x)
zs = getfield.(rbgf_state.particles, :z)
log_ws = getfield.(rbgf_state.particles, :log_w)
ws = softmax(log_ws)

# Reduce tolerance since this is a bit harder to filter to high precision
println("RBGF Outer: ", @test first(kf_state.μ) ≈ sum(only.(xs) .* ws) rtol = 1e-2)
println(
    "RBGF Inner: ", @test last(kf_state.μ) ≈ sum(only.(getfield.(zs, :μ)) .* ws) rtol = 1e-2
)
println("RBGF LL: ", @test llkf ≈ llrbgf atol = 1e-2)
