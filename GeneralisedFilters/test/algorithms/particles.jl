"""Unit tests for particle filter algorithms (non-Rao-Blackwellised)."""

## Bootstrap Filter #########################################################################

@testitem "Bootstrap filter" begin
    using GeneralisedFilters
    using StableRNGs
    using StatsBase: weights

    rng = StableRNG(1234)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(
        rng, 1, 1; static_arrays=true
    )
    _, _, ys = simulate(rng, model, 4)

    resampler = GeneralisedFilters.GFTest.AlternatingResampler()
    bf = BF(10^6; resampler=resampler)
    bf_state, llbf = GeneralisedFilters.filter(rng, model, bf, ys)
    kf_state, llkf = GeneralisedFilters.filter(rng, model, KF(), ys)

    xs = getfield.(bf_state.particles, :state)
    ws = weights(bf_state)

    # Compare log-likelihood and states
    @test first(kf_state.μ) ≈ sum(first.(xs) .* ws) rtol = 1e-3
    @test llkf ≈ llbf atol = 1e-3
end

## Guided Filter ############################################################################

@testitem "Guided filter" begin
    using GeneralisedFilters
    using StableRNGs
    using StatsBase: weights

    rng = StableRNG(1234)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(
        rng, 1, 1; static_arrays=true
    )
    _, _, ys = simulate(rng, model, 4)

    prop = GeneralisedFilters.GFTest.OptimalProposal(model.dyn, model.obs)
    resampler = GeneralisedFilters.GFTest.AlternatingResampler()
    gf = ParticleFilter(10^6, prop; resampler=resampler)
    gf_state, llgf = GeneralisedFilters.filter(rng, model, gf, ys)
    kf_state, llkf = GeneralisedFilters.filter(rng, model, KF(), ys)

    xs = getfield.(gf_state.particles, :state)
    ws = weights(gf_state)

    @test first(kf_state.μ) ≈ sum(first.(xs) .* ws) rtol = 1e-3
    @test llkf ≈ llgf atol = 1e-3
end

## Auxiliary Bootstrap Filter ###############################################################

@testitem "ABF" begin
    using GeneralisedFilters
    using Distributions
    using GeneralisedFilters
    using LinearAlgebra
    using StableRNGs
    using StatsBase: weights

    rng = StableRNG(1234)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(
        rng, 1, 1; static_arrays=true
    )
    _, _, ys = simulate(rng, model, 4)

    resampler = ESSResampler(0.8)
    bf = BF(10^6; resampler=resampler)
    abf = AuxiliaryParticleFilter(bf, MeanPredictive())
    abf_state, llabf = GeneralisedFilters.filter(rng, model, abf, ys)
    kf_state, llkf = GeneralisedFilters.filter(rng, model, KF(), ys)

    xs = getfield.(abf_state.particles, :state)
    ws = weights(abf_state)

    @test first(kf_state.μ) ≈ sum(first.(xs) .* ws) rtol = 1e-2
    @test llkf ≈ llabf atol = 1e-3
end

@testitem "APF evidence correction with nonuniform lookahead" begin
    using GeneralisedFilters
    using LogExpFunctions: logsumexp
    using LinearAlgebra: I
    using Distributions: Bernoulli
    using StableRNGs
    const GF = GeneralisedFilters
    weights = log.([0.2, 0.3, 0.5])
    eta = log.([0.7, 1.8, 0.4])
    likelihood = log.([0.6, 0.2, 0.9])
    particles = [GF.Particle(i, weights[i], i) for i in 1:3]
    state = GF.ParticleDistribution(particles, 0.0)
    # Fixed ancestor draw includes the pinned reference ancestor. Correction must
    # use that ancestor's own lookahead, including for the reference particle.
    idxs = [1, 3, 3]
    resampled = GF.construct_new_state(state, idxs, eta)
    @test GF.log_weights(resampled) ≈ -eta[idxs]
    model = StateSpaceModel(
        DiscretePrior(fill(1 / 3, 3)),
        DiscreteDynamics(Matrix{Float64}(I, 3, 3)),
        DistributionObservation((t, x) -> Bernoulli(exp(likelihood[x]))),
    )
    predicted = GF.predict(StableRNG(13), model.dyn, BF(3), 1, resampled, true)
    _, actual = GF.update(model.obs, BF(3), 1, predicted, true)
    expected =
        logsumexp(weights + eta) - logsumexp(weights) +
        logsumexp(likelihood[idxs] - eta[idxs]) - log(3)
    @test actual ≈ expected
end

@testitem "Population execution preserves scalar hooks and RNG order" begin
    using Random: AbstractRNG, Xoshiro, rand
    using LogExpFunctions: logsumexp
    const GF = GeneralisedFilters

    # An execution request must not silently fall back to CPU particle storage.
    @test_throws ArgumentError initialise(
        Xoshiro(1), GaussianPrior([0.0], [1.0;;]), BF(3; execution=GPUExecution())
    )

    struct HookDynamics <: GF.LatentDynamics end
    struct HookObservation <: GF.ObservationProcess end
    function GF.predict_particle(
        rng::AbstractRNG,
        ::HookDynamics,
        ::GF.BootstrapFilter,
        t::Integer,
        p::GF.Particle,
        y,
        ref,
    )
        return GF.Particle((id=p.state, draw=rand(rng), ref=ref), p.log_w, p.ancestor)
    end
    function GF.update_particle(
        ::HookObservation, ::GF.BootstrapFilter, t::Integer, p::GF.Particle, y
    )
        return GF.Particle((:updated, p.state), p.log_w + p.state.draw, p.ancestor)
    end

    rng, reference_rng = Xoshiro(42), Xoshiro(42)
    draws = [rand(reference_rng) for _ in 1:3]
    initial = GF.ParticleDistribution([GF.Particle(i, 0.0, i) for i in 1:3], 0.0)
    predicted = GF.predict(
        rng, HookDynamics(), BF(3), 1, initial, nothing; ref_state=[98, 99]
    )
    expected = [(id=i, draw=draws[i], ref=i == 1 ? 99 : nothing) for i in 1:3]
    @test getfield.(predicted.particles, :state) == expected
    @test rand(rng) == rand(reference_rng)
    indexed = GF.predict(
        Xoshiro(42),
        HookDynamics(),
        BF(3),
        1,
        initial,
        nothing;
        ref_state=ReferenceTrajectory(98, [99]),
    )
    @test indexed.particles == predicted.particles
    @test_throws ArgumentError GF.predict(
        Xoshiro(42), HookDynamics(), BF(3), 1, initial, nothing; ref_state=[98]
    )
    filtered, _ = GF.update(HookObservation(), BF(3), 1, predicted, nothing)
    @test getfield.(filtered.particles, :state) == [(:updated, p) for p in expected]
    @test GF.log_weights(filtered) ≈ draws .- logsumexp(draws)
end
