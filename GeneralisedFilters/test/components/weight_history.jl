@testitem "CSMC history preserves inferred weight precision" begin
    using GeneralisedFilters
    const GF = GeneralisedFilters

    initial = GF.ParticleDistribution([GF.Particle(i, i) for i in 1:3], GF.TypelessZero())
    for T in (Float32, Float64, BigFloat)
        weights = log.(T[1, 2, 3] ./ T(6))
        first_state = GF.ParticleDistribution(
            [GF.Particle(T(i), weights[i], i) for i in 1:3], zero(T)
        )
        history = GF._init_container(initial, first_state)
        @test eltype(history.weights[1]) === T
        @test history.weights[1] == weights
        @test eltype(history.initial_states) === Int
        @test eltype(history.states[1]) === T
        GF._update_container!(history, first_state)
        @test length(history.weights) == 2
        @test history.weights[2] == weights

        other_type = T === Float64 ? Float32 : Float64
        incompatible = GF.ParticleDistribution(
            [GF.Particle(T(i), other_type(weights[i]), i) for i in 1:3], zero(other_type)
        )
        @test_throws ArgumentError GF._update_container!(history, incompatible)
        @test length(history.weights) == 2
        @test length(history.states) == 2
    end
end

@testitem "APF initial and numeric normalisers retain weight precision" begin
    using GeneralisedFilters, LogExpFunctions, StableRNGs
    const GF = GeneralisedFilters
    indices = [3, 1, 3]
    for T in (Float32, Float64, BigFloat)
        lookahead = T[-0.3, -1.2, -0.7]
        for initial in (true, false)
            old = initial ? [GF.TypelessZero() for _ in 1:3] : log.(T[1, 2, 3] ./ T(6))
            particles = [GF.Particle(i, old[i], i) for i in 1:3]
            state = GF.ParticleDistribution(
                particles, initial ? GF.TypelessZero() : zero(T)
            )
            result = GF.construct_new_state(state, indices, lookahead)
            old_numeric = initial ? zeros(T, 3) : old
            expected = -(
                logsumexp(old_numeric + lookahead) - logsumexp(old_numeric) +
                logsumexp(-lookahead[indices]) - log(T(3))
            )
            @test result.ll_baseline isa T
            @test result.ll_baseline ≈ expected atol=10eps(T)
            @test GF.log_weights(result) == -lookahead[indices]
            @test [p.state for p in result.particles] == indices
        end
    end
    initial = GF.ParticleDistribution([GF.Particle(i, i) for i in 1:3], GF.TypelessZero())
    resampled = GF.resample(StableRNG(131), Multinomial(), initial)
    @test all(p -> p.log_w isa GF.TypelessZero, resampled.particles)
    @test resampled.ll_baseline isa GF.TypelessZero
    @test GF.get_weights(resampled) == fill(1 / 3, 3)
end

@testitem "Float32 CSMC refreshment preserves precision" begin
    using GeneralisedFilters, StableRNGs
    using Distributions: Normal
    const GF = GeneralisedFilters
    model = StateSpaceModel(
        DiscretePrior(Float32[0.6, 0.4]),
        DiscreteDynamics(Float32[0.8 0.2; 0.3 0.7]),
        DistributionObservation((t, x) -> Normal(Float32(x), 0.7f0)),
    )
    observations = Float32[0.2, 1.8, 0.6]
    for strategy in (NoRefreshment(), AncestorSampling(), BackwardSimulation())
        sampler = ConditionalSMC(BF(12; resampler=Multinomial()), strategy)
        rng = StableRNG(713)
        trajectory, ll = GF._csmc_sample(rng, model, sampler, observations, nothing)
        @test ll isa Float32
        @test isfinite(ll)
        @test length(trajectory) == 4
        trajectory, ll = GF._csmc_sample(rng, model, sampler, observations, trajectory)
        @test ll isa Float32
        @test isfinite(ll)
        @test all(x -> x in 1:2, trajectory)
    end
end
