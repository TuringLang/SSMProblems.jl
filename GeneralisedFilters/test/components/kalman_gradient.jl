"""Gradient computation tests for Kalman filter parameters."""

@testitem "Kalman gradient: ∂Q" begin
    using GeneralisedFilters
    using FiniteDifferences
    using LinearAlgebra
    using PDMats
    using StableRNGs
    using StaticArrays
    using SSMProblems

    rng = StableRNG(1234)
    s = GeneralisedFilters.GFTest.setup_gradient_test(rng)
    fdm = central_fdm(5, 1)

    nll_Q = GeneralisedFilters.GFTest.make_nll_func(s.model, s.ys, :Q)
    Q_vec = vec(Matrix(s.Q))
    ∂Q_fd = reshape(FiniteDifferences.grad(fdm, nll_Q, Q_vec)[1], s.D, s.D)
    @test Matrix(s.∂Q_total) ≈ ∂Q_fd rtol = 1e-3  # for some reason this is harder to match
    @test s.Q isa PDMat{Float64,SMatrix{2,2,Float64,4}}
end

@testitem "Kalman gradient: ∂R" begin
    using GeneralisedFilters
    using FiniteDifferences
    using LinearAlgebra
    using PDMats
    using StableRNGs
    using StaticArrays
    using SSMProblems

    rng = StableRNG(1234)
    s = GeneralisedFilters.GFTest.setup_gradient_test(rng)
    fdm = central_fdm(5, 1)

    nll_R = GeneralisedFilters.GFTest.make_nll_func(s.model, s.ys, :R)
    R_vec = vec(Matrix(s.R))
    ∂R_fd = reshape(FiniteDifferences.grad(fdm, nll_R, R_vec)[1], s.D, s.D)
    @test Matrix(s.∂R_total) ≈ ∂R_fd rtol = 1e-4
    @test s.R isa PDMat{Float64,SMatrix{2,2,Float64,4}}
end

@testitem "Kalman gradient: ∂A" begin
    using GeneralisedFilters
    using FiniteDifferences
    using LinearAlgebra
    using PDMats
    using StableRNGs
    using StaticArrays
    using SSMProblems

    rng = StableRNG(1234)
    s = GeneralisedFilters.GFTest.setup_gradient_test(rng)
    fdm = central_fdm(5, 1)

    nll_A = GeneralisedFilters.GFTest.make_nll_func(s.model, s.ys, :A)
    A_vec = vec(Matrix(s.A))
    ∂A_fd = reshape(FiniteDifferences.grad(fdm, nll_A, A_vec)[1], s.D, s.D)
    @test Matrix(s.∂A_total) ≈ ∂A_fd rtol = 1e-4
    @test s.A isa SMatrix{2,2,Float64,4}
end

@testitem "Kalman gradient: ∂b" begin
    using GeneralisedFilters
    using FiniteDifferences
    using LinearAlgebra
    using PDMats
    using StableRNGs
    using StaticArrays
    using SSMProblems

    rng = StableRNG(1234)
    s = GeneralisedFilters.GFTest.setup_gradient_test(rng)
    fdm = central_fdm(5, 1)

    nll_b = GeneralisedFilters.GFTest.make_nll_func(s.model, s.ys, :b)
    ∂b_fd = FiniteDifferences.grad(fdm, nll_b, Vector(s.b))[1]
    @test Vector(s.∂b_total) ≈ ∂b_fd rtol = 1e-4
    @test s.b isa SVector{2,Float64}
end

@testitem "Kalman gradient: ∂μ0" begin
    using GeneralisedFilters
    using FiniteDifferences
    using LinearAlgebra
    using PDMats
    using StableRNGs
    using StaticArrays
    using SSMProblems

    rng = StableRNG(1234)
    s = GeneralisedFilters.GFTest.setup_gradient_test(rng)
    fdm = central_fdm(5, 1)

    nll_μ0 = GeneralisedFilters.GFTest.make_nll_func(s.model, s.ys, :μ0)
    ∂μ0_fd = FiniteDifferences.grad(fdm, nll_μ0, Vector(s.μ0))[1]
    @test Vector(s.∂μ0) ≈ ∂μ0_fd rtol = 1e-4
    @test s.μ0 isa SVector{2,Float64}
end

@testitem "Kalman gradient: ∂Σ0" begin
    using GeneralisedFilters
    using FiniteDifferences
    using LinearAlgebra
    using PDMats
    using StableRNGs
    using StaticArrays
    using SSMProblems

    rng = StableRNG(1234)
    s = GeneralisedFilters.GFTest.setup_gradient_test(rng)
    fdm = central_fdm(5, 1)

    nll_Σ0 = GeneralisedFilters.GFTest.make_nll_func(s.model, s.ys, :Σ0)
    Σ0_vec = vec(Matrix(s.Σ0))
    ∂Σ0_fd = reshape(FiniteDifferences.grad(fdm, nll_Σ0, Σ0_vec)[1], s.D, s.D)
    @test Matrix(s.∂Σ0) ≈ ∂Σ0_fd rtol = 1e-4
    @test s.Σ0 isa PDMat{Float64,SMatrix{2,2,Float64,4}}
end
