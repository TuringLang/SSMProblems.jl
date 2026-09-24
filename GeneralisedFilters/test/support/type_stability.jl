"""Type stability tests using JET.jl."""

@testitem "Kalman filter type stability" begin
    using GeneralisedFilters
    using StableRNGs
    using JET

    const GF = GeneralisedFilters

    rng = StableRNG(1234)
    model = GF.GFTest.create_linear_gaussian_model(rng, 2, 2, Float32; static_arrays=true)
    _, _, ys = GF.simulate(rng, model, 4)
    kf = KalmanFilter()

    # initialise
    @test_opt GF.initialise(rng, model.prior, kf)
    @test_call GF.initialise(rng, model.prior, kf)
    state = GF.initialise(rng, model.prior, kf)

    # predict
    @test_opt GF.predict(rng, model.dyn, kf, 1, state, ys[1])
    @test_call GF.predict(rng, model.dyn, kf, 1, state, ys[1])
    pred = GF.predict(rng, model.dyn, kf, 1, state, ys[1])

    # update
    @test_opt GF.update(model.obs, kf, 1, pred, ys[1])
    @test_call GF.update(model.obs, kf, 1, pred, ys[1])
    _, ll = GF.update(model.obs, kf, 1, pred, ys[1])
    @test ll isa Float32

    # full filtering pass
    @test_opt GF.filter(rng, model, kf, ys)
    @test_call GF.filter(rng, model, kf, ys)
end
