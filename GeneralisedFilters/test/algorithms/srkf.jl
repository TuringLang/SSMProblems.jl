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
