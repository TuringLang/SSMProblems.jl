using Aqua
using Distributions
using LinearAlgebra
using Random
using SSMProblems
using StaticArrays
using Test

## COMPONENTS DEFINED VIA `distribution` ###################################################

struct SimplePrior <: StatePrior end
SSMProblems.distribution(::SimplePrior) = Normal(0, 1)

struct SimpleDynamics{T} <: LatentDynamics
    μ::T
    σ::T
end
function SSMProblems.distribution(dyn::SimpleDynamics, ::Integer, x)
    return Normal(x + dyn.μ, dyn.σ)
end

struct SimpleObservation{T} <: ObservationProcess
    σ::T
end
SSMProblems.distribution(obs::SimpleObservation, ::Integer, x) = Normal(x, obs.σ)

# Controls and parameters belong to the component, not to keyword arguments: a component
# carries whatever it needs and is indexed by the step.
struct SteppedDynamics{T,V} <: LatentDynamics
    μ::T
    σ::T
    dts::V
end
function SSMProblems.distribution(dyn::SteppedDynamics, t::Integer, x)
    dt = dyn.dts[t]
    return Normal(x + dyn.μ * dt, dyn.σ * sqrt(dt))
end

## COMPONENTS IMPLEMENTING `simulate`/`logdensity` DIRECTLY #################################

struct DirectPrior <: StatePrior end
SSMProblems.simulate(::AbstractRNG, ::DirectPrior) = 7.0
SSMProblems.logdensity(::DirectPrior, x0) = -1.5

struct DirectDynamics <: LatentDynamics end
SSMProblems.simulate(::AbstractRNG, ::DirectDynamics, t::Integer, x) = x + t
SSMProblems.logdensity(::DirectDynamics, t::Integer, x_prev, x_new) = -2.5

struct DirectObservation <: ObservationProcess end
SSMProblems.simulate(::AbstractRNG, ::DirectObservation, ::Integer, x) = 2x
SSMProblems.logdensity(::DirectObservation, ::Integer, x, y) = -3.5

## TESTS ###################################################################################

@testset "SSMProblems" begin
    @testset "Distribution-derived simulate and logdensity" begin
        rng = MersenneTwister(1234)
        prior = SimplePrior()
        dyn = SimpleDynamics(0.1, 0.2)
        obs = SimpleObservation(0.3)

        @test simulate(rng, prior) isa Float64
        @test simulate(rng, dyn, 1, 0.5) isa Float64
        @test simulate(rng, obs, 1, 0.5) isa Float64

        @test logdensity(prior, 0.4) ≈ logpdf(Normal(0, 1), 0.4)
        @test logdensity(dyn, 1, 0.5, 0.7) ≈ logpdf(Normal(0.6, 0.2), 0.7)
        @test logdensity(obs, 1, 0.5, 0.2) ≈ logpdf(Normal(0.5, 0.3), 0.2)
    end

    @testset "Directly implemented components" begin
        rng = MersenneTwister(1234)
        @test simulate(rng, DirectPrior()) == 7.0
        @test simulate(rng, DirectDynamics(), 3, 1.0) == 4.0
        @test simulate(rng, DirectObservation(), 1, 2.0) == 4.0
        @test logdensity(DirectPrior(), 0.0) == -1.5
        @test logdensity(DirectDynamics(), 1, 0.0, 0.0) == -2.5
        @test logdensity(DirectObservation(), 1, 0.0, 0.0) == -3.5
    end

    @testset "Forward simulation" begin
        model = StateSpaceModel(
            SimplePrior(), SimpleDynamics(0.1, 0.2), SimpleObservation(0.3)
        )
        T = 3

        x0, xs, ys = simulate(MersenneTwister(1234), model, T)
        @test x0 isa Float64
        @test length(xs) == T
        @test length(ys) == T

        # Reproducible given the same seed, and usable without an explicit rng.
        @test simulate(MersenneTwister(1234), model, T) == (x0, xs, ys)
        @test simulate(model, T) isa Tuple

        # A component carrying step-indexed controls needs no keyword threading.
        stepped = StateSpaceModel(
            SimplePrior(),
            SteppedDynamics(0.1, 0.2, [0.1, 0.2, 0.3]),
            SimpleObservation(0.3),
        )
        @test simulate(MersenneTwister(1234), stepped, T) isa Tuple
    end

    @testset "Forward simulation edge cases" begin
        model = StateSpaceModel(
            SimplePrior(), SimpleDynamics(0.1, 0.2), SimpleObservation(0.3)
        )
        x0, xs, ys = simulate(MersenneTwister(1234), model, 0)
        @test x0 isa Float64
        @test isempty(xs)
        @test isempty(ys)
        @test_throws ArgumentError simulate(MersenneTwister(1234), model, -1)
    end

    @testset "simulate_from_dist preserves static arrays" begin
        rng = MersenneTwister(1234)
        d = MvNormal(SVector(0.0, 1.0), SMatrix{2,2}(1.0, 0.0, 0.0, 1.0))
        x = simulate_from_dist(rng, d)
        @test x isa SVector{2,Float64}

        # The fallback still applies to ordinary distributions.
        @test simulate_from_dist(rng, Normal(0, 1)) isa Float64
        @test simulate_from_dist(rng, MvNormal([0.0, 1.0], I(2))) isa Vector{Float64}
    end

    @testset "Aqua.jl QA" begin
        Aqua.test_all(SSMProblems)
    end
end
