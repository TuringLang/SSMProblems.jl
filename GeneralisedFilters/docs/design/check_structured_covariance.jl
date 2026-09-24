# Standalone design experiment; uses only Julia standard libraries.
# Run: julia --startup-file=no GeneralisedFilters/docs/design/check_structured_covariance.jl
# This verifies algebra, not the production AD integration or numerical conditioning.
using LinearAlgebra
using Random
using Test

sym(A) = (A + A') / 2
coldot(A, B) = vec(sum(A .* B; dims=1))

function check_system(rng, n, m)
    function spd(d)
        X = randn(rng, d, d)
        return X * X' + I
    end
    P0, Q, R = spd(n), spd(n), spd(m)
    A, H = 0.3randn(rng, n, n), randn(rng, m, n)
    x, b, c, y = randn(rng, n), randn(rng, n), randn(rng, m), randn(rng, m)
    a, raw_B, λ = randn(rng, n), randn(rng, n, n), randn(rng)
    B = sym(raw_B)

    function objective(P0, Q, R, A, H, x, b, c, y)
        μ = A * x + b
        P = sym(A * P0 * A' + Q)
        v = y - H * μ - c
        S = sym(H * P * H' + R)
        W = inv(S)
        K = P * H' * W
        J = I - K * H
        Pf = sym(J * P * J' + K * R * K')
        μf = μ + K * v
        ll = -(m * log(2π) + logdet(Symmetric(S)) + dot(v, W * v)) / 2
        return dot(a, μf) + sum(raw_B .* Pf) + λ * ll
    end

    μ = A * x + b
    P = A * P0 * A' + Q
    v = y - H * μ - c
    W = inv(H * P * H' + R)
    K = P * H' * W
    J = I - K * H
    w = W * v
    u, k, r = H' * w, K' * a, J' * a
    E = (λ / 2) * (w * w' - W)
    C = J' * B * J + sym(r * u') + (λ / 2) * (u * u' - H' * W * H)
    D = K' * B * K - sym(k * w') + E

    function apply_C(X)
        JX = X - K * (H * X)
        Z = B * JX
        return Z - H' * (K' * Z) +
               (r * (u' * X) + u * (r' * X)) / 2 +
               (λ / 2) * (u * (u' * X) - H' * (W * (H * X)))
    end
    function apply_D(X)
        return K' * (B * (K * X)) - (k * (w' * X) + w * (k' * X)) / 2 +
               (λ / 2) * (w * (w' * X) - W * X)
    end

    U = B * K
    Z = K' * U
    diag_C =
        diag(B) - 2coldot(U', H) +
        coldot(H, Z * H) +
        r .* u +
        (λ / 2) * (u .^ 2 - coldot(H, W * H))
    diag_D = coldot(K, U) - k .* w + (λ / 2) * (w .^ 2 - diag(W))
    F, Fobs = randn(rng, n, max(1, n - 1)), randn(rng, m, max(1, m - 1))
    for (actual, expected) in (
        (apply_C(F), C * F),
        (apply_D(Fobs), D * Fobs),
        (diag_C, diag(C)),
        (diag_D, diag(D)),
        (apply_C(A), C * A),
    )
        @test isapprox(actual, expected; rtol=1e-10, atol=1e-10)
    end

    ybar = k - λ * w
    μbar = r + λ * u
    T = E - sym(k * w')
    Hbar = -ybar * μ' + w * (P * a)' + 2T * H * P - 2K' * B * J * P
    grads = (
        A' * C * A, C, D, μbar * x' + 2apply_C(A) * P0, Hbar, A' * μbar, μbar, -ybar, ybar
    )
    args = (P0, Q, R, A, H, x, b, c, y)
    for i in eachindex(args)
        direction = randn(rng, size(args[i]))
        ε = 1e-5
        plus = Base.setindex(args, args[i] + ε * direction, i)
        minus = Base.setindex(args, args[i] - ε * direction, i)
        numerical = (objective(plus...) - objective(minus...)) / (2ε)
        analytic = sum(grads[i] .* direction)
        @test isapprox(numerical, analytic; rtol=2e-6, atol=2e-7)
    end

    dF = randn(rng, size(F))
    @test isapprox(
        sum((2apply_C(F)) .* dF), sum(C .* (dF * F' + F * dF')); rtol=1e-10, atol=1e-10
    )
end

@testset "Structured covariance design algebra" begin
    rng = MersenneTwister(42)
    for (n, m) in ((2, 1), (3, 2), (2, 3), (5, 2)), _ in 1:10
        check_system(rng, n, m)
    end
end
