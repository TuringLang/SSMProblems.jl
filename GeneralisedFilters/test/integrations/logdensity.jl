@testitem "trajectory_logdensity and parameter builder contracts" begin
    using Distributions, LogDensityProblems, Random, StaticArrays
    rng = MersenneTwister(16)
    build(θ) = StateSpaceModel(
        GaussianPrior(SVector(θ[1]), SMatrix{1,1}(exp(θ[2]))),
        LinearGaussianDynamics(SMatrix{1,1}(0.8), SVector(θ[1]), SMatrix{1,1}(0.3)),
        LinearGaussianObservation(SMatrix{1,1}(1.0), SVector(0.0), SMatrix{1,1}(0.4)),
    )
    θ = [0.2, -0.3]
    model = build(θ)
    x0, xs, ys = simulate(rng, model, 5)
    ref = ReferenceTrajectory(x0, xs)
    manual =
        logdensity(model.prior, x0) + sum(
            logdensity(model.dyn, t, ref[t - 1], ref[t]) +
            logdensity(model.obs, t, ref[t], ys[t]) for t in 1:5
        )
    @test trajectory_logdensity(model, ref, ys) ≈ manual
    ld = SSMParameterLogDensity(
        MvNormal(zeros(2), ones(2)), ParameterisedSSM(build, ys), ref
    )
    @test LogDensityProblems.dimension(ld) == 2
    @test LogDensityProblems.capabilities(typeof(ld)) ==
        LogDensityProblems.LogDensityOrder{0}()
    @test LogDensityProblems.logdensity(ld, θ) ≈ manual + logpdf(ld.prior, θ)
end

# Replaces the former kf_loglikelihood/ChainRules tests: all eight model parameter
# blocks remain independently exercised through the public shared likelihood evaluator.
@testitem "marginal likelihood all parameter blocks: dense/static forward/reverse" tags = [
    :mooncake
] begin
    using ADTypes,
        ForwardDiff, Mooncake, DifferentiationInterface, StaticArrays, LinearAlgebra
    using FiniteDifferences
    ys = [SVector(0.1, -0.2), SVector(0.3, 0.4), SVector(-0.1, 0.6)]
    θ = [
        0.1,
        -0.1,
        log(0.8),
        log(0.6),
        0.7,
        0.1,
        -0.1,
        0.8,
        0.1,
        0.2,
        log(0.3),
        log(0.2),
        1.0,
        0.1,
        0.2,
        0.9,
        0.1,
        -0.1,
        log(0.4),
        log(0.5),
    ]
    function objective(θ, static)
        v(x) = static ? SVector{2}(x) : Vector(x)
        m(x) = static ? SMatrix{2,2}(x) : Matrix(x)
        prior = GaussianPrior(v(θ[1:2]), m(Diagonal(exp.(θ[3:4]))))
        dyn = LinearGaussianDynamics(
            m(reshape(θ[5:8], 2, 2)), v(θ[9:10]), m(Diagonal(exp.(θ[11:12])))
        )
        obs = LinearGaussianObservation(
            m(reshape(θ[13:16], 2, 2)), v(θ[17:18]), m(Diagonal(exp.(θ[19:20])))
        )
        return marginal_loglikelihood(StateSpaceModel(prior, dyn, obs), KF(), ys)
    end
    @test objective(θ, true) ≈ objective(θ, false)
    for static in (false, true)
        f = θ -> objective(θ, static)
        fd = FiniteDifferences.grad(central_fdm(5, 1), f, θ)[1]
        fw = ForwardDiff.gradient(f, θ)
        rv = DifferentiationInterface.gradient(f, AutoMooncake(; config=nothing), θ)
        @test fw ≈ fd atol = 1e-7 rtol = 1e-6
        @test rv ≈ fd atol = 1e-7 rtol = 1e-6
    end
end

@testitem "hierarchical full density uses shared conditional likelihood" begin
    using Distributions, StaticArrays, LogDensityProblems
    build(θ) = StateSpaceModel(
        GaussianPrior(SVector(θ[1]), SMatrix{1,1}(1.0)),
        LinearGaussianDynamics(SMatrix{1,1}(0.8), SVector(θ[1]), SMatrix{1,1}(0.2)),
        ctx -> GaussianPrior(SVector(θ[1] + ctx.x0[1]), SMatrix{1,1}(0.5)),
        ctx -> LinearGaussianDynamics(
            SMatrix{1,1}(0.7), SVector(θ[1] + ctx.x_new[1]), SMatrix{1,1}(0.3)
        ),
        ctx -> LinearGaussianObservation(
            SMatrix{1,1}(1.0), SVector(ctx.x[1]), SMatrix{1,1}(0.4)
        ),
    )
    θ = [0.2]
    xs = [SVector(0.1), SVector(0.3), SVector(-0.1)]
    ys = [SVector(0.4), SVector(0.1)]
    m = build(θ)
    inner = condition_inner(m, xs)
    _, ll = GeneralisedFilters.filter(inner, KF(), ys)
    outer =
        logdensity(m.prior.outer, xs[1]) +
        sum(logdensity(m.dyn.outer, t, xs[t], xs[t + 1]) for t in 1:2)
    @test trajectory_logdensity(m, KF(), xs, ys) ≈ outer + ll
    ld = SSMParameterLogDensity(
        MvNormal([0.0], [1.0]), ParameterisedSSM(build, ys), KF(), xs
    )
    @test LogDensityProblems.logdensity(ld, θ) ≈ logpdf(ld.prior, θ) + outer + ll
end
