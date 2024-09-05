using Distributions
using Random
using SSMProblems
using Test

@testset "Test forward simulation" begin
    struct TestLatentDynamics <: LatentDynamics{Float64}
        μ::Float64
        σ::Float64
    end
    SSMProblems.distribution(dyn::TestLatentDynamics, extra) = Normal(0, 1)
    SSMProblems.distribution(dyn::TestLatentDynamics, step::Integer, prev_state, extra) =
        Normal(prev_state + dyn.μ, dyn.σ)

    struct TestObservationProcess <: ObservationProcess{Float64}
        σ::Float64
    end
    SSMProblems.distribution(obs::TestObservationProcess, step::Integer, state, extra) =
        Normal(state, obs.σ)

    model = StateSpaceModel(TestLatentDynamics(0.1, 0.2), TestObservationProcess(0.3))

    rng = MersenneTwister(1234)
    T = 3
    extra0 = nothing
    extras = [nothing for _ in 1:T]

    # Sampling with/without rng and extras
    @test sample(rng, model, extra0, extras) isa Tuple
    @test sample(rng, model, T) isa Tuple
    @test sample(model, extra0, extras) isa Tuple
    @test sample(model, T) isa Tuple
end
