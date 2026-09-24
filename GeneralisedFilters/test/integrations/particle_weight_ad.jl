@testitem "Particle weights: AD through a fixed cloud without resampling" tags = [:mooncake] begin
    using GeneralisedFilters, Distributions, Random, ForwardDiff, Mooncake
    using LogExpFunctions: logsumexp
    const GF = GeneralisedFilters

    struct WeightADTransition{T} <: LatentDynamics
        shift::T
    end
    GF.simulate(rng::AbstractRNG, d::WeightADTransition, t::Integer, x) = x + d.shift

    # Different fixed particles make normalisation and accumulation nontrivial. The
    # transition changes ordinary initial states into Dual states on the first step.
    function objective(θ)
        initial = GF.ParticleDistribution(
            [GF.Particle(-1.0f0, 1), GF.Particle(2.0f0, 2)], GF.TypelessZero()
        )
        model = StateSpaceModel(
            DistributionPrior(Dirac(0.0f0)),
            WeightADTransition(θ[1]),
            DistributionObservation((t, x) -> Normal(x, exp(θ[2]))),
        )
        pf = BF(2; threshold=0.0)
        rng = Xoshiro(42)
        state, ll = GF.step(rng, model, pf, 1, initial, 0.3f0)
        state, inc = GF.step(rng, model, pf, 2, state, -0.2f0)
        return ll + inc
    end
    function reference(θ)
        scores = map((-1.0f0, 2.0f0)) do x
            return logpdf(Normal(x + θ[1], exp(θ[2])), 0.3f0) +
                   logpdf(Normal(x + 2θ[1], exp(θ[2])), -0.2f0)
        end
        return logsumexp(collect(scores)) - log(oftype(θ[1], 2))
    end
    for θ in (Float32[0.1, -0.2], [0.1, -0.2])
        @test typeof(objective(θ)) === eltype(θ)
        @test objective(θ) ≈ reference(θ)
        forward = ForwardDiff.gradient(objective, θ)
        @test forward ≈ ForwardDiff.gradient(reference, θ) rtol = 3e-5
        @test ForwardDiff.hessian(objective, θ) ≈ ForwardDiff.hessian(reference, θ) rtol =
            3e-5
        cache = Mooncake.prepare_gradient_cache(objective, θ)
        value, (_, reverse) = Mooncake.value_and_gradient!!(cache, objective, θ)
        @test value ≈ reference(θ)
        @test reverse ≈ forward rtol = 3e-5
        θnew = θ .+ eltype(θ)(0.05)
        value, (_, reverse) = Mooncake.value_and_gradient!!(cache, objective, θnew)
        @test value ≈ reference(θnew)
        @test reverse ≈ ForwardDiff.gradient(reference, θnew) rtol = 3e-5
    end
    initial = GF.ParticleDistribution(
        [GF.Particle(-1.0f0, 1), GF.Particle(2.0f0, 2)], GF.TypelessZero()
    )
    changing = StateSpaceModel(
        DistributionPrior(Dirac(0.0f0)),
        WeightADTransition(0.1f0),
        DistributionObservation(
            (t, x) -> t == 1 ? Normal(x, 1.0f0) : Normal(Float64(x), 1.0)
        ),
    )
    rng = Xoshiro(43)
    pf = BF(2; threshold=0.0)
    first_state, _ = GF.step(rng, changing, pf, 1, initial, 0.3f0)
    @test eltype(GF.log_weights(first_state)) === Float32
    @test_throws ArgumentError GF.step(rng, changing, pf, 2, first_state, 0.2f0)
end

@testitem "Public particle filter retains natural precision under AD" tags = [:mooncake] begin
    using GeneralisedFilters, Distributions, Random, ForwardDiff, Mooncake
    function objective(θ)
        model = StateSpaceModel(
            DistributionPrior(Dirac(0.0f0)),
            DistributionDynamics((t, x) -> Dirac(x + θ[1])),
            DistributionObservation((t, x) -> Normal(x, exp(θ[2]))),
        )
        _, ll = GeneralisedFilters.filter(
            Xoshiro(42), model, BF(2; threshold=0.0), Float32[0.3, -0.2]
        )
        return ll
    end
    reference(θ) =
        logpdf(Normal(θ[1], exp(θ[2])), 0.3f0) + logpdf(Normal(2θ[1], exp(θ[2])), -0.2f0)
    for θ in (Float32[0.1, -0.2], [0.1, -0.2])
        @test typeof(objective(θ)) === eltype(θ)
        @test objective(θ) ≈ reference(θ)
        cache = Mooncake.prepare_gradient_cache(objective, θ)
        value, (_, reverse) = Mooncake.value_and_gradient!!(cache, objective, θ)
        @test value ≈ reference(θ)
        @test reverse ≈ ForwardDiff.gradient(reference, θ) rtol = 3e-5
        @test ForwardDiff.gradient(objective, θ) ≈ reverse rtol = 3e-5
    end
end
