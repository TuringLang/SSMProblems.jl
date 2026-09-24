"""Reverse-mode gradient tests for the Kalman stack against finite differences."""

@testitem "AD: homogeneous linear-Gaussian marginal likelihood" tags = [:mooncake] begin
    using GeneralisedFilters
    using GeneralisedFilters.GFTest: check_gradients
    using StaticArrays
    using StableRNGs
    using Mooncake

    rng = StableRNG(1234)
    ys = [SVector{2}(randn(rng, 2)) for _ in 1:8]
    μ0, Σ0 = SA[0.0, 0.0], SA[1.0 0.0; 0.0 1.0]
    A, b = SA[0.9 0.1; 0.0 0.8], SA[0.05, -0.02]
    H, c = SA[1.0 0.0; 0.0 1.0], SA[0.0, 0.0]

    function build(θ)
        Q = exp(θ[1]) * SA[1.0 0.0; 0.0 1.0]
        R = exp(θ[2]) * SA[1.0 0.0; 0.0 1.0]
        return StateSpaceModel(
            GaussianPrior(μ0, Σ0),
            LinearGaussianDynamics(A, b, Q),
            LinearGaussianObservation(H, c, R),
        )
    end
    nll(θ) = -marginal_loglikelihood(build(θ), KalmanFilter(), ys)

    @test check_gradients(nll, [-0.5, 0.3]).agrees
end

@testitem "AD: time-varying linear-Gaussian marginal likelihood" tags = [:mooncake] begin
    using GeneralisedFilters
    using GeneralisedFilters.GFTest: check_gradients
    using StaticArrays
    using StableRNGs
    using Mooncake

    rng = StableRNG(11)
    T = 10
    dts = [1.0 + 0.5sin(0.5t) for t in 1:T]
    ys = [SVector{1}(randn(rng, 1)) for _ in 1:T]

    function build(θ)
        q = exp(θ[1])
        R = exp(θ[2]) * SMatrix{1,1}(1.0)
        function dyn((; t))
            dt = dts[t]
            A = SA[1.0 dt; 0.0 1.0]
            Q = q * SA[dt^3/3 dt^2/2; dt^2/2 dt]
            return LinearGaussianDynamics(A, SA[0.0, 0.0], Q)
        end
        return StateSpaceModel(
            GaussianPrior(SA[0.0, 0.0], SA[1.0 0.0; 0.0 1.0]),
            TimeVaryingDynamics(dyn),
            LinearGaussianObservation(SA[1.0 0.0], SA[0.0], R),
        )
    end
    nll(θ) = -marginal_loglikelihood(build(θ), KalmanFilter(), ys)

    @test check_gradients(nll, [-0.7, 0.2]).agrees
end

@testitem "AD: conditionally linear-Gaussian inner likelihood and activity flags" tags = [
    :mooncake
] begin
    using GeneralisedFilters
    using GeneralisedFilters: resolve, kalman_step, with_activity
    using GeneralisedFilters.GFTest: check_gradients
    using StaticArrays
    using StableRNGs
    using Distributions
    using Mooncake

    rng = StableRNG(7)
    T = 12
    dts = [1.0 + 0.3cos(0.4t) for t in 1:T]
    outer = [0.2 * randn(rng) for _ in 0:T]          # fixed outer trajectory x0..xT
    ys = [SVector{1}(randn(rng, 1)) for _ in 1:T]

    # θ enters a hoisted parameter (Q, R) and a shared θ/outer precompute in A; b depends on
    # both x_prev and x_new but not θ, so it is inactive.
    function build(θ)
        a, logq, logr = θ[1], θ[2], θ[3]
        Q = exp(logq) * SA[1.0 0.0; 0.0 1.0]
        R = exp(logr) * SMatrix{1,1}(1.0)
        function inner_dyn((; t, x_prev, x_new))
            s = a * x_new
            A = exp(s) * SA[0.5 0.05; 0.0 0.5]
            b = SA[dts[t] * x_prev, x_new]
            return LinearGaussianDynamics(A, b, Q)
        end
        return StateSpaceModel(
            HierarchicalPrior(
                DistributionPrior(Normal(0.0, 1.0)),
                GaussianPrior(SA[0.0, 0.0], SA[1.0 0.0; 0.0 1.0]),
            ),
            HierarchicalDynamics(
                DistributionDynamics((t, x) -> Normal(0.9x, 0.3)), inner_dyn
            ),
            HierarchicalObservation(LinearGaussianObservation(SA[1.0 0.0], SA[0.0], R)),
        )
    end

    function inner_ll(model, xs, ys)
        p = resolve(model.prior.inner, (; x0=xs[1]))
        state = GaussianState(p.μ0, p.Σ0)
        ll = zero(eltype(p.μ0))
        for t in eachindex(ys)
            d = resolve(model.dyn.inner, (; t, x_prev=xs[t], x_new=xs[t + 1]))
            o = resolve(model.obs.inner, (; t, x=xs[t + 1]))
            state, inc = kalman_step(state, d, o, ys[t])
            ll += inc
        end
        return ll
    end

    θ0 = [0.4, -0.5, 0.2]
    nll(θ) = -inner_ll(build(θ), outer, ys)
    active = check_gradients(nll, θ0)
    @test active.agrees

    # Stamping the true activity pattern skips inactive adjoints without changing the result.
    flags = Val((dyn=(true, false, true), obs=(false, false, true)))
    nll_flagged(θ) = -inner_ll(with_activity(build(θ), flags), outer, ys)
    flagged = check_gradients(nll_flagged, θ0)
    @test flagged.agrees
    @test flagged.ad ≈ active.ad
end

@testitem "AD: inactive field adjoints are exactly zero" begin
    using GeneralisedFilters
    using GeneralisedFilters:
        kalman_step_cached,
        _kalman_reverse_core,
        _A_adjoint,
        _b_adjoint,
        _Q_adjoint,
        _H_adjoint,
        _c_adjoint,
        _R_adjoint
    using StaticArrays

    state = GaussianState(SA[0.1, 0.2], SA[1.0 0.0; 0.0 1.0])
    d = LinearGaussianDynamics(SA[0.9 0.1; 0.0 0.8], SA[0.0, 0.0], SA[0.1 0.0; 0.0 0.1])
    o = LinearGaussianObservation(SA[1.0 0.0], SA[0.0], SMatrix{1,1}(0.5))
    _, _, c = kalman_step_cached(state, d, o, SA[0.3])
    g = _kalman_reverse_core(c, SA[1.0, 0.5], SA[0.1 0.0; 0.0 0.2], 1.0)

    @test _A_adjoint(Val(false), c, g) == zero(c.A)
    @test _b_adjoint(Val(false), c, g) == zero(g.μ̂̄)
    @test _Q_adjoint(Val(false), c, g) == zero(g.Σ̂̄)
    @test _H_adjoint(Val(false), c, g) == zero(c.H)
    @test _c_adjoint(Val(false), c, g) == zero(g.ŷ̄)
    @test _R_adjoint(Val(false), c, g) == zero(g.S̄)

    @test _A_adjoint(Val(true), c, g) == _A_adjoint(c, g)
    @test _R_adjoint(Val(true), c, g) == _R_adjoint(c, g)
end

@testitem "AD: eigenvalue-clipping repair rrule" tags = [:mooncake] begin
    using GeneralisedFilters
    using GeneralisedFilters: repair_covariance, EigenClip
    using GeneralisedFilters.GFTest: check_gradients
    using StaticArrays
    using Mooncake

    ε = 0.5
    Σbase = SA[1.0 0.3; 0.3 0.2]                       # smallest eigenvalue below ε
    W = SA[0.7 0.1; 0.1 1.3]
    function f(θ)
        M = θ[1] * Σbase + θ[2] * SA[0.0 0.1; 0.1 0.0]
        Σ = (M + M') / 2
        return sum(W .* repair_covariance(EigenClip(ε), Σ))
    end

    @test check_gradients(f, [1.2, 0.3]; rtol=1e-5).agrees
end

@testitem "AD: ForwardDiff differentiates the Kalman likelihood" begin
    using GeneralisedFilters
    using GeneralisedFilters.GFTest: central_diff
    using StaticArrays
    using StableRNGs
    using ForwardDiff

    rng = StableRNG(3)
    ys = [SVector{2}(randn(rng, 2)) for _ in 1:6]
    μ0, Σ0 = SA[0.0, 0.0], SA[1.0 0.0; 0.0 1.0]
    A, b = SA[0.9 0.1; 0.0 0.8], SA[0.0, 0.0]
    H, c = SA[1.0 0.0; 0.0 1.0], SA[0.0, 0.0]

    function build(θ)
        Q = exp(θ[1]) * SA[1.0 0.0; 0.0 1.0]
        R = exp(θ[2]) * SA[1.0 0.0; 0.0 1.0]
        return StateSpaceModel(
            GaussianPrior(μ0, Σ0),
            LinearGaussianDynamics(A, b, Q),
            LinearGaussianObservation(H, c, R),
        )
    end
    nll(θ) = -marginal_loglikelihood(build(θ), KalmanFilter(), ys)
    θ0 = [-0.3, 0.4]

    @test ForwardDiff.gradient(nll, θ0) ≈ central_diff(nll, θ0) rtol = 1e-6
end

@testitem "AD: parameter-dependent observations" tags = [:mooncake] begin
    using GeneralisedFilters
    using GeneralisedFilters.GFTest: check_gradients
    using StaticArrays
    using Mooncake
    using ForwardDiff
    model = create_homogeneous_linear_gaussian_model(
        SA[0.1],
        SMatrix{1,1}(1.0),
        SMatrix{1,1}(0.8),
        SA[0.0],
        SMatrix{1,1}(0.2),
        SMatrix{1,1}(1.0),
        SA[0.0],
        SMatrix{1,1}(0.3),
    )
    f(θ) = marginal_loglikelihood(model, KF(), [SA[θ[1]], SA[θ[1] + θ[2]]])
    result = check_gradients(f, [0.2, -0.4])
    @test result.agrees
    @test result.ad ≈ ForwardDiff.gradient(f, [0.2, -0.4])
end

@testitem "AD: clipping threshold and unconstrained matrix storage" tags = [:mooncake] begin
    using GeneralisedFilters
    using GeneralisedFilters: repair_covariance
    using GeneralisedFilters.GFTest: check_gradients
    using StaticArrays
    using Mooncake
    # Lower entries are ignored by Symmetric; clipping threshold is differentiable
    # away from eigenvalue crossings. Check each independently parameterised entry.
    f(θ) = sum(
        SA[0.2 0.3; 0.4 0.7] .*
        repair_covariance(EigenClip(θ[5]), SA[θ[1] θ[2]; θ[3] θ[4]]),
    )
    @test check_gradients(f, [1.0, 0.1, -0.2, 0.2, 0.5]).agrees
    # A near-degenerate pair on opposite sides of the threshold still has a
    # nontrivial divided difference; it cannot be replaced by a midpoint derivative.
    ext = Base.get_extension(GeneralisedFilters, :MooncakeExt)
    K = ext._clip_divided_differences(SA[0.5 - 1e-10, 0.5 + 1e-10], 0.5)
    @test K[1, 2] ≈ 0.5
end

@testitem "AD: mixed precision uses output cotangent precision" tags = [:mooncake] begin
    using GeneralisedFilters
    using GeneralisedFilters: kalman_step
    using GeneralisedFilters.GFTest: check_gradients
    using StaticArrays
    using Mooncake
    state = GaussianState(SA[0.1f0], SMatrix{1,1}(1.0f0))
    obs = LinearGaussianObservation(SMatrix{1,1}(1.0), SA[0.0], SMatrix{1,1}(0.3))
    function f(θ)
        dyn = LinearGaussianDynamics(SMatrix{1,1}(θ[1]), SA[0.2], SMatrix{1,1}(0.1))
        filtered, ll = kalman_step(state, dyn, obs, SA[0.4])
        return 1e40 * filtered.μ[1] + ll
    end
    result = check_gradients(f, [0.8])
    @test all(isfinite, result.ad)
    @test result.agrees
end
