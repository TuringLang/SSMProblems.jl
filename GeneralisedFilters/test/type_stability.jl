# using JET
@testitem "Particle Filter type stability" begin
    using GeneralisedFilters
    using SSMProblems
    using StableRNGs
    using JET

    const GF = GeneralisedFilters

    rng = StableRNG(1234)
    model = GF.GFTest.create_linear_gaussian_model(rng, 1, 1, Float32; static_arrays=true)
    _, _, ys = sample(rng, model, 4)
    algo = BF(2^4)

    # initialize
    @test_call GF.initialise(rng, prior(model), algo)
    init_state = GF.initialise(rng, prior(model), algo)

    # resample
    rs = GF.resampler(algo)
    @test_call GF.maybe_resample(rng, rs, init_state)
    state = GF.maybe_resample(rng, rs, init_state)

    # predict
    @test_call GF.predict(rng, dyn(model), algo, 1, state, ys[1])
    state = GF.predict(rng, dyn(model), algo, 1, state, ys[1])

    # update
    @test_call GF.update(obs(model), algo, 1, state, ys[1])
    _, ll = GF.update(obs(model), algo, 1, state, ys[1])
    @test ll isa Float32
end
