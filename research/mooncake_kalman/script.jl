# Mooncake Kalman Filter Gradient Benchmark
#
# Compares gradient computation for the Kalman filter log-likelihood using:
# 1. Zygote (uses ChainRules rrule from kalman_rrule.jl)
# 2. Mooncake (uses native rrule!! from kalman_mooncake.jl)
# 3. Mooncake auto-diff (no custom rule, for comparison)

using GeneralisedFilters
using Distributions
using PDMats
using LinearAlgebra
using Random

using DifferentiationInterface
using Mooncake: Mooncake
using Zygote: Zygote
using BenchmarkTools

const GF = GeneralisedFilters

## MODEL PARAMETERS ###########################################################################

const Dx, Dy = 2, 2
const T_len = 100

rng = MersenneTwister(1234)

const μ0_fixed = randn(rng, Dx)
const Σ0_fixed = let M = randn(rng, Dx, Dx)
    PDMat(Symmetric(M * M' + 0.1I))
end
const A_template = randn(rng, Dx, Dx)
const b_fixed = randn(rng, Dx)
const Q_fixed = let M = randn(rng, Dx, Dx)
    PDMat(Symmetric(M * M' + 0.1I))
end
const H_fixed = randn(rng, Dy, Dx)
const c_fixed = randn(rng, Dy)
const R_fixed = let M = randn(rng, Dy, Dy)
    PDMat(Symmetric(M * M' + 0.1I))
end
const ys_fixed = [randn(rng, Dy) for _ in 1:T_len]

## AUTO-DIFF MODULE (NO RULE) #################################################################

# Separate module to bypass the package's rrule for auto-diff comparison
module NoRuleTest
using GeneralisedFilters
using Distributions

const GF = GeneralisedFilters

function kf_ll_direct(μ0, Σ0, As, bs, Qs, Hs, cs, Rs, ys)
    state = MvNormal(μ0, Σ0)
    ll = zero(eltype(μ0))
    for t in eachindex(ys)
        state = GF.kalman_predict(state, (As[t], bs[t], Qs[t]))
        state, ll_inc = GF.kalman_update(state, (Hs[t], cs[t], Rs[t]), ys[t], nothing)
        ll += ll_inc
    end
    return ll
end
end

## OBJECTIVE FUNCTIONS ########################################################################

function make_params(θ::AbstractVector{T}) where {T}
    A = T.(A_template) .* θ[1]
    return (
        T.(μ0_fixed),
        Σ0_fixed,
        fill(A, T_len),
        fill(T.(b_fixed), T_len),
        fill(Q_fixed, T_len),
        fill(T.(H_fixed), T_len),
        fill(T.(c_fixed), T_len),
        fill(R_fixed, T_len),
        ys_fixed,
    )
end

logℓ(θ) = GF.kf_loglikelihood(make_params(θ)...)
logℓ_no_rule(θ) = NoRuleTest.kf_ll_direct(make_params(θ)...)

## BENCHMARK SETUP ############################################################################

println("=" ^ 70)
println("Kalman Filter Gradient Benchmark: Zygote vs Mooncake")
println("=" ^ 70)
println()
println("Model: $(Dx)D state, $(Dy)D observation, $(T_len) timesteps")
println()

θ_init = [0.5]

println("Verifying gradient correctness...")
println("-" ^ 70)

grad_zygote = Zygote.gradient(logℓ, θ_init)[1]
println("Zygote (ChainRules rrule):        ", grad_zygote)

grad_mooncake = DifferentiationInterface.gradient(logℓ, AutoMooncake(; config=nothing), θ_init)
println("Mooncake (native rrule!!):        ", grad_mooncake)

grad_auto = nothing
auto_diff_works = false
try
    global grad_auto, auto_diff_works
    grad_auto = DifferentiationInterface.gradient(
        logℓ_no_rule, AutoMooncake(; config=nothing), θ_init
    )
    println("Mooncake (auto-diff, no rule):    ", grad_auto)
    auto_diff_works = true
catch e
    println("Mooncake (auto-diff, no rule):    FAILED - ", typeof(e))
end

all_match = isapprox(grad_zygote, grad_mooncake; rtol=1e-6) &&
            (!auto_diff_works || isapprox(grad_zygote, grad_auto; rtol=1e-6))
if all_match
    println("✓ All computed gradients match!")
end
println()

## BENCHMARKS #################################################################################

println("Running benchmarks...")
println("-" ^ 70)
println()

println("1. Zygote (ChainRules rrule):")
zygote_bench = @benchmark Zygote.gradient($logℓ, $θ_init)
display(zygote_bench)
println()

println("2. Mooncake (native rrule!!):")
mooncake_prep = DifferentiationInterface.prepare_gradient(
    logℓ, AutoMooncake(; config=nothing), θ_init
)
mooncake_bench = @benchmark DifferentiationInterface.gradient(
    $logℓ, $mooncake_prep, AutoMooncake(; config=nothing), $θ_init
)
display(mooncake_bench)
println()

auto_bench = nothing
if auto_diff_works
    println("3. Mooncake (auto-diff, no rule):")
    auto_prep = DifferentiationInterface.prepare_gradient(
        logℓ_no_rule, AutoMooncake(; config=nothing), θ_init
    )
    auto_bench = @benchmark DifferentiationInterface.gradient(
        $logℓ_no_rule, $auto_prep, AutoMooncake(; config=nothing), $θ_init
    )
    display(auto_bench)
    println()
end

## SUMMARY ####################################################################################

println("=" ^ 70)
println("Summary")
println("=" ^ 70)
println()

zygote_ms = median(zygote_bench.times) / 1e6
mooncake_ms = median(mooncake_bench.times) / 1e6

println("Median times:")
println("  Zygote (ChainRules rrule):     $(round(zygote_ms; digits=3)) ms")
println("  Mooncake (native rrule!!):     $(round(mooncake_ms; digits=3)) ms")
if auto_diff_works && auto_bench !== nothing
    auto_ms = median(auto_bench.times) / 1e6
    println("  Mooncake (auto-diff):          $(round(auto_ms; digits=3)) ms")
end

println()
println("Speedup vs Zygote:")
println("  Mooncake rrule: $(round(zygote_ms / mooncake_ms; digits=2))x")
if auto_diff_works && auto_bench !== nothing
    println("  Mooncake auto:  $(round(zygote_ms / (median(auto_bench.times)/1e6); digits=2))x")
end
