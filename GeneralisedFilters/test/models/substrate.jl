@testitem "Shared SSMProblems interface" begin
    using SSMProblems
    using GeneralisedFilters
    using Random
    using Distributions

    for name in (
        :StatePrior,
        :LatentDynamics,
        :ObservationProcess,
        :StateSpaceModel,
        :distribution,
        :simulate,
        :logdensity,
        :simulate_from_dist,
    )
        @test getproperty(GeneralisedFilters, name) === getproperty(SSMProblems, name)
    end

    # Components defined entirely against the substrate need no GF-specific methods.
    struct SubstratePrior <: SSMProblems.StatePrior end
    struct SubstrateDynamics <: SSMProblems.LatentDynamics end
    struct SubstrateObservation <: SSMProblems.ObservationProcess end
    SSMProblems.simulate(::AbstractRNG, ::SubstratePrior) = 0.0
    SSMProblems.simulate(::AbstractRNG, ::SubstrateDynamics, t::Integer, x) = x + 1
    SSMProblems.distribution(::SubstrateObservation, t::Integer, x) = Normal(x, 0.5)

    model = SSMProblems.StateSpaceModel(
        SubstratePrior(), SubstrateDynamics(), SubstrateObservation()
    )
    rng = MersenneTwister(12)
    x0, xs, ys = SSMProblems.simulate(rng, model, 4)
    @test x0 == 0.0
    @test xs == collect(1.0:4.0)
    _, ll = GeneralisedFilters.filter(rng, model, BF(8), ys)
    @test ll ≈ sum(logpdf(Normal(x, 0.5), y) for (x, y) in zip(xs, ys))
end
