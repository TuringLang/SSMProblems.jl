using Distributions
using Random
using Aqua
using SSMProblems
using Test

@testset "Forward Simulation" begin
    @testset "Forward simulation without control" begin
        struct UncontrolledPrior <: StatePrior end
        SSMProblems.distribution(::UncontrolledPrior; kwargs...) = Normal(0, 1)

        struct UncontrolledLatentDynamics{T} <: LatentDynamics
            μ::T
            σ::T
        end
        function SSMProblems.distribution(
            dyn::UncontrolledLatentDynamics, ::Integer, prev_state; kwargs...
        )
            return Normal(prev_state + dyn.μ, dyn.σ)
        end

        struct UncontrolledObservationProcess{T} <: ObservationProcess
            σ::T
        end
        function SSMProblems.distribution(
            obs::UncontrolledObservationProcess, ::Integer, state; kwargs...
        )
            return Normal(state, obs.σ)
        end

        model = StateSpaceModel(
            UncontrolledPrior(),
            UncontrolledLatentDynamics(0.1, 0.2),
            UncontrolledObservationProcess(0.3)
        )

        rng = MersenneTwister(1234)
        T = 3

        # Sampling with/without rng
        @test sample(rng, model, T) isa Tuple
        @test sample(model, T) isa Tuple
    end

    @testset "Forward simulation with control" begin
        struct ControlledPrior <: StatePrior end
        SSMProblems.distribution(::ControlledPrior; σ_init, kwargs...) = Normal(0, σ_init)

        struct ControlledLatentDynamics{T} <: LatentDynamics
            μ::T
            σ::T
        end
        function SSMProblems.distribution(
            dyn::ControlledLatentDynamics, step::Integer, prev_state; dts, kwargs...
        )
            dt = dts[step]
            return Normal(prev_state + dyn.μ * dt, dyn.σ * sqrt(dt))
        end

        struct ControlledObservationProcess{T} <: ObservationProcess
            σ::T
        end
        function SSMProblems.distribution(
            obs::ControlledObservationProcess, ::Integer, state; kwargs...
        )
            return Normal(state, obs.σ)
        end

        model = StateSpaceModel(
            ControlledPrior(),
            ControlledLatentDynamics(0.1, 0.2),
            ControlledObservationProcess(0.3)
        )

        rng = MersenneTwister(1234)
        T = 3

        σ_init = 0.5
        dts = [0.1, 0.2, 0.3]

        # Sampling with/without rng
        @test sample(rng, model, T; σ_init=σ_init, dts=dts) isa Tuple
        @test sample(model, T; σ_init=σ_init, dts=dts) isa Tuple
    end
end

@testset "Aqua.jl QA" begin
    Aqua.test_all(SSMProblems; deps_compat=false)
end
