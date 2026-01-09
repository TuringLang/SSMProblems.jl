"""Unit tests for particle filter algorithms (non-Rao-Blackwellised)."""

## Bootstrap Filter #########################################################################

@testitem "Bootstrap filter" begin
    using SSMProblems
    using StableRNGs
    using StatsBase: weights

    rng = StableRNG(1234)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(
        rng, 1, 1; static_arrays=true
    )
    _, _, ys = sample(rng, model, 4)

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
    using SSMProblems
    using StableRNGs
    using StatsBase: weights

    rng = StableRNG(1234)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(
        rng, 1, 1; static_arrays=true
    )
    _, _, ys = sample(rng, model, 4)

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
    using Distributions
    using GeneralisedFilters
    using LinearAlgebra
    using SSMProblems
    using StableRNGs
    using StatsBase: weights

    rng = StableRNG(1234)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(
        rng, 1, 1; static_arrays=true
    )
    _, _, ys = sample(rng, model, 4)

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
