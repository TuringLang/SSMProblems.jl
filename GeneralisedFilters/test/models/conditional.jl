@testitem "Conditional model: Gaussian joint likelihood and ordinary filter reuse" begin
    using GeneralisedFilters
    using GeneralisedFilters: resolve
    using Distributions
    using StaticArrays
    using LinearAlgebra

    xs = [0.4, -0.3, 0.7]
    model = StateSpaceModel(
        DistributionPrior(Normal()),
        DistributionDynamics((t, x) -> Normal(0.8x, 0.5)),
        (((; x0),) -> GaussianPrior(SA[2x0], SMatrix{1,1}(1.5))),
        (
            ((; t, x_prev, x_new),) -> LinearGaussianDynamics(
                SMatrix{1,1}(0.4 + 0.1x_prev + 0.2x_new + 0.03t),
                SA[0.1x_prev],
                SMatrix{1,1}(0.2),
            )
        ),
        (
            ((; t, x),) -> LinearGaussianObservation(
                SMatrix{1,1}(1 + 0.1x), SA[0.2t], SMatrix{1,1}(0.3)
            )
        ),
    )
    inner = @inferred condition_inner(model, xs)
    reference = ReferenceTrajectory(xs[1], xs[2:end])
    from_reference = @inferred condition_inner(model, reference)
    ys = [SA[0.25], SA[-0.15]]

    # Independent two-observation joint distribution, integrating out z0,z1,z2.
    a1 = 0.4 + 0.1xs[1] + 0.2xs[2] + 0.03
    a2 = 0.4 + 0.1xs[2] + 0.2xs[3] + 0.06
    h1, h2 = 1 + 0.1xs[2], 1 + 0.1xs[3]
    m1 = a1 * (2xs[1]) + 0.1xs[1]
    m2 = a2 * m1 + 0.1xs[2]
    p1 = a1^2 * 1.5 + 0.2
    p2 = a2^2 * p1 + 0.2
    expected = logpdf(
        MvNormal(
            [h1 * m1 + 0.2, h2 * m2 + 0.4],
            [h1^2 * p1+0.3 h1*h2*a2*p1; h1*h2*a2*p1 h2^2 * p2+0.3],
        ),
        [0.25, -0.15],
    )

    @test inner isa StateSpaceModel
    @test inner.prior.μ0 isa SVector
    @test resolve(inner.dyn, (; t=1)).A isa SMatrix
    @test marginal_loglikelihood(inner, KF(), ys) ≈ expected
    @test marginal_loglikelihood(from_reference, KF(), ys) ≈ expected
    @test inner_loglikelihood(KF(), model, xs, ys) ≈ expected
    final_state, ll = GeneralisedFilters.filter(inner, KF(), ys)
    @test final_state.μ isa SVector
    @test final_state.Σ isa SMatrix
    @test ll ≈ expected
    smoothed, smooth_ll = smooth(inner, KS, ys)
    @test smooth_ll ≈ expected
    @test smoothed.μ isa SVector
    flagged = condition_inner(
        with_activity(model, Val((dyn=(true, true, true), obs=(true, true, true)))),
        reference,
    )
    @test last(GeneralisedFilters.filter(flagged, KF(), ys)) ≈ expected
    @test last(smooth(flagged, KS, ys)) ≈ expected

    expected_outer =
        logpdf(Normal(), xs[1]) +
        logpdf(Normal(0.8xs[1], 0.5), xs[2]) +
        logpdf(Normal(0.8xs[2], 0.5), xs[3])
    @test outer_logdensity(model, reference) ≈ expected_outer
    @test trajectory_logdensity(model, KF(), reference, ys) ≈ expected_outer + expected

    # The same conditional parameters define the generative joint model.
    s0 = HierarchicalState(xs[1], SA[0.1])
    s1 = HierarchicalState(xs[2], SA[-0.2])
    @test logdensity(model.prior, s0) ≈
        logpdf(Normal(), xs[1]) + logdensity(inner.prior, s0.z)
    @test logdensity(model.dyn, 1, s0, s1) ≈
        logpdf(Normal(0.8xs[1], 0.5), xs[2]) + logdensity(inner.dyn, 1, s0.z, s1.z)
    @test logdensity(model.obs, 1, s1, ys[1]) ≈ logdensity(inner.obs, 1, s1.z, ys[1])
end

@testitem "Conditional model: trajectory lengths and initial-only objective" begin
    using GeneralisedFilters
    using GeneralisedFilters: resolve
    using Distributions
    using StaticArrays

    model = StateSpaceModel(
        DistributionPrior(Normal()),
        DistributionDynamics((t, x) -> Normal(x, 1.0)),
        (((; x0),) -> GaussianPrior(SA[x0], SMatrix{1,1}(1.0))),
        LinearGaussianDynamics(SMatrix{1,1}(0.8), SA[0.0], SMatrix{1,1}(0.2)),
        LinearGaussianObservation(SMatrix{1,1}(1.0), SA[0.0], SMatrix{1,1}(0.3)),
    )
    @test_throws ArgumentError condition_inner(model, Float64[])
    xs = ReferenceTrajectory(0.4, [0.2, -0.1])
    inner = condition_inner(model, xs)
    @test_throws DimensionMismatch marginal_loglikelihood(inner, KF(), [SA[0.0]])
    @test_throws DimensionMismatch marginal_loglikelihood(inner, KF(), fill(SA[0.0], 3))
    @test_throws ArgumentError resolve(inner.dyn, (; t=3))
    @test_throws ArgumentError marginal_loglikelihood(
        inner, KF(), ReferenceTrajectory(SA[0.0], [SA[0.0]])
    )
    ys0 = SVector{1,Float64}[]
    for initial_only in ([0.4], ReferenceTrajectory(0.4, Float64[]))
        @test_throws ArgumentError marginal_loglikelihood(
            condition_inner(model, initial_only), KF(), ys0
        )
        @test_throws ArgumentError trajectory_logdensity(model, KF(), initial_only, ys0)
    end
end

@testitem "Filtering rejects empty data and zero-step simulation avoids transitions" begin
    using GeneralisedFilters
    using StaticArrays
    using Random
    p = GaussianPrior(SA[0.0], SMatrix{1,1}(1.0))
    d = TimeVaryingDynamics(ctx -> error("no transition should be evaluated"))
    o = TimeVaryingObservation(ctx -> error("no observation should be evaluated"))
    model = StateSpaceModel(p, d, o)
    for af in (KF(), SRKF(), BF(2))
        @test_throws ArgumentError GeneralisedFilters.filter(
            model, af, SVector{1,Float64}[]
        )
    end
    x0, xs, ys = simulate(MersenneTwister(2), model, 0)
    @test x0 isa SVector{1,Float64}
    @test isempty(xs) && isempty(ys)
    @test_throws ArgumentError simulate(model, -1)
    @test_throws ArgumentError smooth(model, KS, ys)
end
