"""Tests for Mooncake.jl integration with kf_loglikelihood."""

@testitem "kf_loglikelihood Mooncake: dense arrays" tags = [:mooncake] begin
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

@testitem "kf_loglikelihood Mooncake: StaticArrays" tags = [:mooncake] begin
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

@testitem "kf_loglikelihood Mooncake: PDiagMat" tags = [:mooncake] begin
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

@testitem "kf_loglikelihood Mooncake: multiple parameters" tags = [:mooncake] begin
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
@testitem "kf_loglikelihood Mooncake: accumulation and edge cases" tags = [:mooncake] begin
    using GeneralisedFilters,
        Mooncake, PDMats, LinearAlgebra, StaticArrays, FiniteDifferences

    μ = [0.1]
    A, H = reshape([0.8], 1, 1), ones(1, 1)
    P, Q, R = PDMat(1.7H), PDMat(0.5H), PDMat(0.3H)
    ys = [[0.4], [-0.7]]
    function ll(m; bs=fill([0.05], 2), data=ys, jitter=nothing)
        return kf_loglikelihood(
            m,
            P,
            fill(A, 2),
            bs,
            fill(Q, 2),
            fill(H, 2),
            fill([0.0], 2),
            fill(R, 2),
            data,
            jitter,
        )
    end
    cases = (
        ("external contribution", x -> ll(x) + sum(x), μ),
        ("two likelihoods", x -> ll(x) + ll(x), μ),
        ("shared mean and offsets", x -> ll(x; bs=fill(x, 2)), μ),
        ("observations", x -> ll(μ; data=[x, ys[2]]), ys[1]),
        ("fixed jitter", x -> ll(x; jitter=0.01), μ),
        ("view mean", x -> ll(view(x, 1:1)), μ),
        ("static time container", x -> ll(μ; bs=SVector(x, x)), μ),
        (
            "immutable time and element containers",
            x -> ll(μ; bs=SVector(SVector{1}(x), SVector{1}(x))),
            μ,
        ),
        ("view time container", x -> ll(μ; bs=view([x, x], :)), μ),
        ("empty sequence", x -> ll(x; data=Vector{eltype(x)}[]), μ),
    )
    @testset "$name" for (name, f, x) in cases
        x = copy(x) # Finite differences must not perturb captured constants.
        expected = only(FiniteDifferences.grad(central_fdm(5, 1), f, x))
        cache = Mooncake.prepare_gradient_cache(f, x)
        for _ in 1:2
            value, (_, actual) = Mooncake.value_and_gradient!!(cache, f, x)
            @test value ≈ f(x)
            @test actual ≈ expected atol=1e-8 rtol=1e-6
        end
    end
    f(x) = ll(μ; jitter=only(x))
    cache = Mooncake.prepare_gradient_cache(f, [0.01])
    _, (_, grad) = Mooncake.value_and_gradient!!(cache, f, [0.01])
    @test iszero(grad) # Jitter is a nondifferentiable constant.
end

@testitem "kf_loglikelihood Mooncake: covariance representations" tags = [:mooncake] begin
    using GeneralisedFilters,
        Mooncake, PDMats, LinearAlgebra, StaticArrays, FiniteDifferences

    μ = [0.1, -0.2]
    A, H = [0.7 0.1; -0.2 0.8], [1.0 0.2; 0.1 0.9]
    P = PDMat([1.7 0.0; 0.0 1.2])
    Q, R = PDMat([0.5 0.0; 0.0 0.4]), PDMat([0.3 0.0; 0.0 0.2])
    ys = [[0.4, -0.7], [0.2, 0.8]]
    wrappers = (
        ("matrix", x -> diagm(x), [1.7, 1.2]),
        ("diagonal", x -> Diagonal(x), [1.7, 1.2]),
        ("PDiagMat", x -> PDiagMat(x), [1.7, 1.2]),
        ("ScalMat", x -> ScalMat(2, only(x)), [1.7]),
        ("PDMat diagonal", x -> PDMat(Diagonal(x)), [1.7, 1.2]),
        (
            "static matrix, dense factor",
            x -> PDMat(SMatrix{2,2}(diagm(x)), cholesky(diagm(x))),
            [1.7, 1.2],
        ),
        (
            "dense matrix, static factor",
            x -> PDMat(diagm(x), cholesky(SMatrix{2,2}(diagm(x)))),
            [1.7, 1.2],
        ),
    )
    @testset "$name $role" for (name, wrap, x) in wrappers,
        role in
        (wrap(x) isa AbstractPDMat ? (:prior, :Q, :R, :shared) : (:prior, :Q, :shared))

        function f(x)
            covariance = wrap(x)
            return kf_loglikelihood(
                μ,
                role in (:prior, :shared) ? covariance : P,
                fill(A, 2),
                fill(μ, 2),
                fill(role in (:Q, :shared) ? covariance : Q, 2),
                fill(H, 2),
                fill(μ, 2),
                fill(role == :R ? covariance : R, 2),
                ys,
            )
        end
        expected = only(FiniteDifferences.grad(central_fdm(5, 1), f, x))
        cache = Mooncake.prepare_gradient_cache(f, x)
        value, (_, actual) = Mooncake.value_and_gradient!!(cache, f, x)
        @test value ≈ f(x)
        @test actual ≈ expected atol=1e-8 rtol=1e-6
    end
    @testset "$F $static $uplo $wrapper" for F in (Float32, Float64),
        static in (false, true), uplo in (:U, :L),
        wrapper in (Symmetric, Hermitian)

        mat(x) = static ? SMatrix{2,2}(x) : Matrix(x)
        vec(x) = static ? SVector{2}(x) : Vector(x)
        function f(x)
            S = wrapper(mat(reshape(x, 2, 2)), uplo)
            # StaticArrays cholesky ignores :L; factor the materialized matrix.
            covariance = PDMat(S, cholesky(mat(S)))
            m, a, h = vec(F.(μ)), mat(F.(A)), mat(F.(H))
            return kf_loglikelihood(
                m,
                covariance,
                fill(a, 2),
                fill(m, 2),
                fill(covariance, 2),
                fill(h, 2),
                fill(m, 2),
                fill(covariance, 2),
                [vec(F.(y)) for y in ys],
            )
        end
        x = F[1.7, 0.1, 0.3, 1.2]
        expected = only(FiniteDifferences.grad(central_fdm(5, 1), f, x))
        cache = Mooncake.prepare_gradient_cache(f, x)
        value, (_, actual) = Mooncake.value_and_gradient!!(cache, f, x)
        @test value ≈ f(x)
        @test actual ≈ expected atol=1e-5 rtol=1e-3
        @test actual[uplo == :U ? 2 : 3] == 0
    end
end
