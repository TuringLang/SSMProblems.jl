@testitem "Custom model containers across inference boundaries" begin
    using SSMProblems, GeneralisedFilters, StaticArrays, Random, Distributions, ForwardDiff
    const GF = GeneralisedFilters
    struct ComponentModel{T} <: AbstractStateSpaceModel
        parts::T
    end
    SSMProblems.prior(m::ComponentModel) = m.parts[1]
    SSMProblems.dyn(m::ComponentModel) = m.parts[2]
    SSMProblems.obs(m::ComponentModel) = m.parts[3]
    wrap(m) = ComponentModel((prior(m), dyn(m), obs(m)))
    matrix(x) = SMatrix{1,1}(x)
    build(a) = StateSpaceModel(
        GaussianPrior(SA[0.0], matrix(1.0)),
        TimeVaryingDynamics(
            ctx -> LinearGaussianDynamics(matrix(a), SA[0.1ctx.t], matrix(0.3))
        ),
        LinearGaussianObservation(matrix(1.0), SA[0.0], matrix(0.2)),
    )
    model = build(0.7)
    custom = wrap(model)
    ys = [SA[0.1], SA[-0.2]]
    @test simulate(MersenneTwister(7), custom, 2) == simulate(MersenneTwister(7), model, 2)
    for algo in (KF(), SRKF(), BF(12))
        @test last(GF.filter(MersenneTwister(1), custom, algo, ys)) ≈
            last(GF.filter(MersenneTwister(1), model, algo, ys))
    end
    @test last(smooth(custom, KS, ys)) ≈ last(smooth(model, KS, ys))
    @test marginal_loglikelihood(custom, KF(), ys) ≈ marginal_loglikelihood(model, KF(), ys)
    @test ForwardDiff.derivative(
        a -> marginal_loglikelihood(wrap(build(a)), KF(), ys), 0.7
    ) ≈ ForwardDiff.derivative(a -> marginal_loglikelihood(build(a), KF(), ys), 0.7)
    @test logpdf(SSMTrajectory(custom, ys), [0.0, 0.1, 0.2]) ≈
        logpdf(SSMTrajectory(model, ys), [0.0, 0.1, 0.2])

    hier = StateSpaceModel(
        prior(model),
        dyn(model),
        ctx -> GaussianPrior(SA[ctx.x0[1]], matrix(1.0)),
        ctx ->
            LinearGaussianDynamics(matrix(0.8), SA[ctx.x_new[1] + 0.1ctx.t], matrix(0.3)),
        ctx -> LinearGaussianObservation(matrix(1.0), SA[0.1ctx.t], matrix(0.2)),
    )
    hcustom = wrap(hier)
    xs = [SA[0.0], SA[0.1], SA[0.2]]
    @test marginal_loglikelihood(condition_inner(hcustom, xs), KF(), ys) ≈
        inner_loglikelihood(KF(), hier, xs, ys)
    @test trajectory_logdensity(hcustom, KF(), xs, ys) ≈
        trajectory_logdensity(hier, KF(), xs, ys)
    @test logpdf(SSMTrajectory(hcustom, KF(), ys), [0.0, 0.1, 0.2]) ≈
        trajectory_logdensity(hier, KF(), xs, ys)
    @test last(GF.filter(MersenneTwister(1), hcustom, RBPF(BF(8), KF()), ys)) ≈
        last(GF.filter(MersenneTwister(1), hier, RBPF(BF(8), KF()), ys))
    for strategy in (NoRefreshment(), AncestorSampling(), BackwardSimulation())
        sampler = ConditionalSMC(RBPF(BF(8), KF()), strategy)
        ref = ReferenceTrajectory(xs[1], xs[2:end])
        a, ll_a = GF._csmc_sample(MersenneTwister(2), hcustom, sampler, ys, ref)
        b, ll_b = GF._csmc_sample(MersenneTwister(2), hier, sampler, ys, ref)
        @test collect(a) == collect(b)
        @test ll_a ≈ ll_b
    end
end

@testitem "Component factory errors explain the required interface" begin
    using GeneralisedFilters, Distributions, StaticArrays
    const GF = GeneralisedFilters
    @test_throws ArgumentError GF.resolve(TimeVaryingDynamics(ctx -> Normal()), (; t=1))
    @test_throws ArgumentError GF.resolve(TimeVaryingObservation(ctx -> Normal()), (; t=1))
    @test_throws ArgumentError inner_prior(ctx -> Normal(), 0.0)
    @test_throws ArgumentError inner_dynamics(ctx -> Normal(), 1, 0.0, 0.0)
    @test_throws ArgumentError inner_observation(ctx -> Normal(), 1, 0.0)
    @test_throws DomainError GF.resolve(
        TimeVaryingDynamics(ctx -> throw(DomainError(ctx.t))), (; t=1)
    )
    model = StateSpaceModel(
        GaussianPrior(SA[0.0], SMatrix{1,1}(1.0)),
        DistributionDynamics((t, x) -> MvNormal(x, SMatrix{1,1}(1.0))),
        LinearGaussianObservation(SMatrix{1,1}(1.0), SA[0.0], SMatrix{1,1}(1.0)),
    )
    @test_throws ArgumentError GF.filter(model, KF(), [SA[0.0]])
end
