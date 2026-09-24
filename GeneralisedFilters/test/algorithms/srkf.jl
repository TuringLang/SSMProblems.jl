"""Unit tests for Square Root Kalman filter."""

@testitem "SRKF filter" begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs

    SEED = 1234
    Dx = 3
    Dys = [2, 3, 4]
    T = 5

    for Dy in Dys
        rng = StableRNG(SEED)
        model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)
        _, _, ys = simulate(rng, model, T)

        kf_state, kf_ll = GeneralisedFilters.filter(StableRNG(SEED), model, KF(), ys)
        srkf_sqrt, srkf_ll = GeneralisedFilters.filter(StableRNG(SEED), model, SRKF(), ys)
        srkf_state = GaussianState(srkf_sqrt)

        @test srkf_state.μ ≈ kf_state.μ
        @test srkf_state.Σ ≈ kf_state.Σ
        @test srkf_ll ≈ kf_ll
    end
end

@testitem "SRKF filter StaticArrays" begin
    using GeneralisedFilters
    using StableRNGs
    using StaticArrays

    D = 2
    rng = StableRNG(1234)

    μ0 = @SVector rand(rng, D)
    Σ0 = @SMatrix rand(rng, D, D)
    Σ0 = Σ0 * Σ0'
    A = @SMatrix rand(rng, D, D)
    b = @SVector rand(rng, D)
    Q = @SMatrix rand(rng, D, D)
    Q = Q * Q'
    H = @SMatrix rand(rng, D, D)
    c = @SVector rand(rng, D)
    R = @SMatrix rand(rng, D, D)
    R = R * R'

    model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)

    _, _, ys = simulate(rng, model, 2)

    sqrt_state, _ = GeneralisedFilters.filter(rng, model, SRKF(), ys)
    state = GaussianState(sqrt_state)

    @test ys[2] isa SVector{D,Float64}
    @test state.μ isa SVector{D,Float64}
    @test state.Σ isa SMatrix{D,D,Float64}
end

@testitem "Square-root marginal likelihood and conditional objective" begin
    using GeneralisedFilters, StaticArrays, ForwardDiff
    p = GaussianPrior(SVector(0.0), SMatrix{1,1}(1.0))
    d = LinearGaussianDynamics(SMatrix{1,1}(0.8), SVector(0.1), SMatrix{1,1}(0.3))
    observation(c) =
        LinearGaussianObservation(SMatrix{1,1}(1.0), SVector(c), SMatrix{1,1}(0.4))
    ys = [SVector(0.1), SVector(-0.3)]
    model = StateSpaceModel(p, d, observation(0.0))
    @test marginal_loglikelihood(model, SRKF(), ys) ≈
        marginal_loglikelihood(model, KF(), ys)
    @test_throws ArgumentError marginal_loglikelihood(model, SRKF(), SVector{1,Float64}[])
    f(c, af) = marginal_loglikelihood(StateSpaceModel(p, d, observation(c)), af, ys)
    @test ForwardDiff.derivative(c -> f(c, SRKF()), 0.2) ≈
        ForwardDiff.derivative(c -> f(c, KF()), 0.2)
    hier = StateSpaceModel(p, d, p, d, observation(0.0))
    path = ReferenceTrajectory(SVector(0.2), [SVector(0.3), SVector(-0.1)])
    @test trajectory_logdensity(hier, SRKF(), path, ys) ≈
        trajectory_logdensity(hier, KF(), path, ys)
end
