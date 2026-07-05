"""Unit tests for the GaussianState type and covariance helpers."""

@testitem "GaussianState rand and logpdf" begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StaticArrays
    using StableRNGs

    rng = StableRNG(1234)

    μ = [0.5, -1.0, 2.0]
    L = randn(rng, 3, 3)
    Σ = L * L' + I

    g = GaussianState(μ, Σ)
    mvn = MvNormal(μ, Symmetric(Σ))

    # rand from a plain-vector mean returns a Vector of the right length
    x = rand(StableRNG(1), g)
    @test x isa Vector{Float64}
    @test length(x) == 3

    # logpdf matches the equivalent Distributions.MvNormal
    for _ in 1:5
        z = randn(rng, 3)
        @test logpdf(g, z) ≈ logpdf(mvn, z)
    end

    # boundary interop
    @test MvNormal(g) isa MvNormal
    @test mean(g) == μ
    @test cov(g) == Σ
    @test length(g) == 3
end

@testitem "GaussianState StaticArrays" begin
    using GeneralisedFilters
    using StaticArrays
    using StableRNGs

    rng = StableRNG(1234)
    μ = @SVector [1.0, 2.0]
    Σ = SMatrix{2,2}(2.0, 0.3, 0.3, 1.0)

    g = GaussianState(μ, Σ)
    x = rand(rng, g)

    # A static mean yields a static draw
    @test x isa SVector{2,Float64}
    @test eltype(g) == Float64
end

@testitem "symmetrise" begin
    using GeneralisedFilters: symmetrise
    using LinearAlgebra
    using StaticArrays

    A = [1.0 2.0; 0.0 3.0]
    S = symmetrise(A)
    @test S == S'
    @test S ≈ (A + A') / 2

    # symmetric input is left unchanged and static type is preserved
    B = SMatrix{2,2}(4.0, 1.0, 1.0, 5.0)
    SB = symmetrise(B)
    @test SB isa SMatrix
    @test SB ≈ B
end

@testitem "GaussianState isapprox" begin
    using GeneralisedFilters
    using StaticArrays

    g1 = GaussianState(SA[1.0, 2.0], SA[1.0 0.0; 0.0 2.0])
    g2 = GaussianState(SA[1.0 + 1e-12, 2.0], SA[1.0 0.0; 0.0 2.0 + 1e-12])
    g3 = GaussianState(SA[1.5, 2.0], SA[1.0 0.0; 0.0 2.0])
    g4 = GaussianState(SA[1.0, 2.0], SA[1.0 0.0; 0.0 2.5])

    @test g1 ≈ g2
    @test !(g1 ≈ g3)  # differs in mean
    @test !(g1 ≈ g4)  # differs in covariance
end

@testitem "SqrtGaussianState round-trip" begin
    using GeneralisedFilters
    using LinearAlgebra
    using StaticArrays
    using StableRNGs

    rng = StableRNG(1234)
    μ = @SVector [1.0, -2.0]
    L = @SMatrix randn(rng, 2, 2)
    Σ = L * L' + I

    g = GaussianState(μ, Σ)
    sq = GeneralisedFilters.SqrtGaussianState(g)
    g2 = GaussianState(sq)

    @test g2.μ == μ
    @test g2.Σ ≈ Σ
end
