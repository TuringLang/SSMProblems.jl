"""Tests for the Mooncake `rrule!!` on `_kalman_step`.

A single Kalman step is differentiated via Mooncake; the gradient w.r.t. the dynamics
offset `b` is compared against finite differences. Wraps the call in a closure that
returns just `ll`, exercising the rrule's tuple-output pullback with zero cotangents on
the state outputs.
"""

@testitem "_kalman_step Mooncake rrule: ∂b (StaticArrays)" tags = [:mooncake] begin
    using GeneralisedFilters
    using GeneralisedFilters: _kalman_step
    using Mooncake
    using FiniteDifferences
    using StableRNGs
    using PDMats
    using StaticArrays
    using LinearAlgebra

    rng = StableRNG(1234)
    Dx, Dy = 2, 2

    μ_prev = @SVector randn(rng, Dx)
    Σ_prev = let M = @SMatrix randn(rng, Dx, Dx)
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
    y = @SVector randn(rng, Dy)

    function step_ll(b_vec)
        b_s = SVector{Dx}(b_vec)
        _, _, ll = _kalman_step(μ_prev, Σ_prev, A, b_s, Q, H, c, R, y, nothing)
        return ll
    end

    b_vec = Vector(b)
    fdm = central_fdm(5, 1)
    grad_fd = FiniteDifferences.grad(fdm, step_ll, b_vec)[1]
    cache = Mooncake.prepare_gradient_cache(step_ll, b_vec)
    _, (_, grad_mooncake) = Mooncake.value_and_gradient!!(cache, step_ll, b_vec)

    @test grad_mooncake ≈ grad_fd rtol = 1e-5
end
