"""Tests for Mooncake.jl integration with kf_loglikelihood."""

@testitem "kf_loglikelihood Mooncake: dense arrays" begin
    using GeneralisedFilters
    using Mooncake
    using Zygote
    using StableRNGs
    using PDMats
    using LinearAlgebra

    rng = StableRNG(1234)
    Dx, Dy, T = 2, 2, 3
    μ0, Σ0, A, b, Q, H, c, R, ys = GeneralisedFilters.GFTest.setup_kf_rrule_params(
        rng, Dx, Dy, T
    )

    As = fill(A, T)
    bs = fill(b, T)
    Qs = fill(Q, T)
    Hs = fill(H, T)
    cs = fill(c, T)
    Rs = fill(R, T)

    # Function that varies b
    function ll_b(b_vec)
        return kf_loglikelihood(μ0, Σ0, As, fill(b_vec, T), Qs, Hs, cs, Rs, ys)
    end

    # Compare Mooncake to Zygote
    grad_zygote = Zygote.gradient(ll_b, b)[1]
    cache = Mooncake.prepare_gradient_cache(ll_b, b)
    _, (_, grad_mooncake) = Mooncake.value_and_gradient!!(cache, ll_b, b)

    @test grad_mooncake ≈ grad_zygote rtol = 1e-6
end

@testitem "kf_loglikelihood Mooncake: StaticArrays" begin
    using GeneralisedFilters
    using Mooncake
    using Zygote
    using StableRNGs
    using PDMats
    using StaticArrays
    using LinearAlgebra

    rng = StableRNG(1234)
    Dx, Dy, T = 2, 2, 3

    # Build static array parameters
    μ0 = @SVector randn(rng, Dx)
    Σ0 = let M = @SMatrix randn(rng, Dx, Dx)
        PDMat(Symmetric(M * M' + 0.1I))
    end
    A = @SMatrix randn(rng, Dx, Dx)
    b = @SVector randn(rng, Dx)
    Q = let M = @SMatrix randn(rng, Dx, Dx)
        PDMat(Symmetric(M * M' + 0.1I))
    end
    H = @SMatrix randn(rng, Dy, Dx)
    c = @SVector randn(rng, Dy)
    R = let M = @SMatrix randn(rng, Dy, Dy)
        PDMat(Symmetric(M * M' + 0.1I))
    end

    As = fill(A, T)
    Qs = fill(Q, T)
    Hs = fill(H, T)
    cs = fill(c, T)
    Rs = fill(R, T)
    ys = [SVector{Dy}(randn(rng, Dy)) for _ in 1:T]

    # Function that varies b
    function ll_b(b_vec)
        b_s = SVector{Dx}(b_vec)
        return kf_loglikelihood(μ0, Σ0, As, fill(b_s, T), Qs, Hs, cs, Rs, ys)
    end

    # Compare Mooncake to Zygote (using Vector input for compatibility)
    b_vec = Vector(b)
    grad_zygote = Zygote.gradient(ll_b, b_vec)[1]
    cache = Mooncake.prepare_gradient_cache(ll_b, b_vec)
    _, (_, grad_mooncake) = Mooncake.value_and_gradient!!(cache, ll_b, b_vec)

    @test grad_mooncake ≈ grad_zygote rtol = 1e-6
end

@testitem "kf_loglikelihood Mooncake: PDiagMat" begin
    using GeneralisedFilters
    using Mooncake
    using FiniteDifferences
    using StableRNGs
    using PDMats
    using StaticArrays
    using LinearAlgebra

    rng = StableRNG(1234)
    Dx, Dy, T = 2, 2, 3

    # Build parameters with PDiagMat for R
    μ0 = @SVector randn(rng, Dx)
    Σ0 = let M = @SMatrix randn(rng, Dx, Dx)
        PDMat(Symmetric(M * M' + 0.1I))
    end
    A = @SMatrix randn(rng, Dx, Dx)
    b = @SVector randn(rng, Dx)
    Q = let M = @SMatrix randn(rng, Dx, Dx)
        PDMat(Symmetric(M * M' + 0.1I))
    end
    H = @SMatrix randn(rng, Dy, Dx)
    c = @SVector randn(rng, Dy)

    # Use PDiagMat for observation noise
    r_diag = SVector{Dy}(abs.(randn(rng, Dy)) .+ 0.1)
    R = PDiagMat(r_diag)

    As = fill(A, T)
    bs = fill(b, T)
    Qs = fill(Q, T)
    Hs = fill(H, T)
    cs = fill(c, T)
    Rs = fill(R, T)
    ys = [SVector{Dy}(randn(rng, Dy)) for _ in 1:T]

    # Function that varies R diagonal
    function ll_r(r_vec)
        R_new = PDiagMat(SVector{Dy}(r_vec))
        return kf_loglikelihood(μ0, Σ0, As, bs, Qs, Hs, cs, fill(R_new, T), ys)
    end

    # Mooncake gradient
    r_vec = Vector(r_diag)
    cache = Mooncake.prepare_gradient_cache(ll_r, r_vec)
    _, (_, grad_mooncake) = Mooncake.value_and_gradient!!(cache, ll_r, r_vec)

    # Finite differences for ground truth
    fdm = central_fdm(5, 1)
    grad_fd = FiniteDifferences.grad(fdm, ll_r, r_vec)[1]

    @test grad_mooncake ≈ grad_fd rtol = 1e-4
end

@testitem "kf_loglikelihood Mooncake: multiple parameters" begin
    using GeneralisedFilters
    using Mooncake
    using Zygote
    using StableRNGs
    using PDMats
    using LinearAlgebra

    rng = StableRNG(1234)
    Dx, Dy, T = 2, 2, 3
    μ0, Σ0, A, b, Q, H, c, R, ys = GeneralisedFilters.GFTest.setup_kf_rrule_params(
        rng, Dx, Dy, T
    )

    As = fill(A, T)
    Qs = fill(Q, T)
    Hs = fill(H, T)
    Rs = fill(R, T)

    # Function that varies both b and c
    function ll_bc(b_vec, c_vec)
        return kf_loglikelihood(μ0, Σ0, As, fill(b_vec, T), Qs, Hs, fill(c_vec, T), Rs, ys)
    end

    # Compare gradients
    grad_zygote = Zygote.gradient(ll_bc, b, c)
    cache = Mooncake.prepare_gradient_cache(ll_bc, b, c)
    _, (_, grad_b, grad_c) = Mooncake.value_and_gradient!!(cache, ll_bc, b, c)

    @test grad_b ≈ grad_zygote[1] rtol = 1e-6
    @test grad_c ≈ grad_zygote[2] rtol = 1e-6
end
