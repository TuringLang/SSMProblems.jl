"""Tests for the log-density interface (trajectory_logdensity, ssm_loglikelihood)."""

## Regular SSM trajectory_logdensity ###########################################################

@testitem "trajectory_logdensity: regular SSM" begin
    using GeneralisedFilters
    using SSMProblems
    using StableRNGs
    using Distributions
    using GeneralisedFilters: ReferenceTrajectory

    let
        rng = StableRNG(1234)
        Dx, Dy, T = 2, 2, 5
        model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)

        x0, xs, ys = SSMProblems.sample(rng, model, T)
        trajectory = ReferenceTrajectory(x0, xs)

        ll = trajectory_logdensity(model, trajectory, ys)

        ll_manual = logpdf(SSMProblems.distribution(SSMProblems.prior(model)), x0)
        for t in 1:T
            ll_manual += SSMProblems.logdensity(
                SSMProblems.dyn(model), t, trajectory[t - 1], trajectory[t]
            )
            ll_manual += SSMProblems.logdensity(
                SSMProblems.obs(model), t, trajectory[t], ys[t]
            )
        end

        @test ll ≈ ll_manual
    end
end

## HierarchicalSSM trajectory_logdensity #######################################################

@testitem "trajectory_logdensity: HierarchicalSSM" begin
    using GeneralisedFilters
    using SSMProblems
    using StableRNGs
    using Distributions
    using GeneralisedFilters: ReferenceTrajectory

    let
        rng = StableRNG(1234)
        D_outer, D_inner, D_obs, T = 2, 2, 2, 5

        full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
            rng, D_outer, D_inner, D_obs; static_arrays=false
        )
        x0, z0, xs, zs, ys = SSMProblems.sample(rng, hier_model, T)
        outer_traj = ReferenceTrajectory(x0, xs)

        ll = trajectory_logdensity(hier_model, KF(), outer_traj, ys)

        ll_manual = logpdf(SSMProblems.distribution(hier_model.outer_prior), outer_traj[0])
        for t in 1:T
            ll_manual += SSMProblems.logdensity(
                hier_model.outer_dyn, t, outer_traj[t - 1], outer_traj[t]
            )
        end

        inner_model = hier_model.inner_model
        state = GeneralisedFilters.initialise(
            rng, inner_model.prior, KF(); new_outer=outer_traj[0]
        )
        ll_inner = 0.0
        for t in 1:T
            state = GeneralisedFilters.predict(
                rng,
                inner_model.dyn,
                KF(),
                t,
                state,
                nothing;
                prev_outer=outer_traj[t - 1],
                new_outer=outer_traj[t],
            )
            state, ll_inc = GeneralisedFilters.update(
                inner_model.obs, KF(), t, state, ys[t]; new_outer=outer_traj[t]
            )
            ll_inner += ll_inc
        end
        ll_manual += ll_inner

        @test ll ≈ ll_manual
    end
end

## TrajectoryParameterLogDensity ###############################################################

@testitem "TrajectoryParameterLogDensity: regular SSM" begin
    using GeneralisedFilters
    using GeneralisedFilters: FixedParametric, ReferenceTrajectory
    using SSMProblems
    using LogDensityProblems
    using StableRNGs
    using Distributions
    using PDMats
    using LinearAlgebra

    rng = StableRNG(1234)

    # Simple 1D model with parametric drift b
    a = 0.8
    q² = 0.1
    r² = 0.5
    σ₀² = 1.0
    T_len = 5

    prior_ssm = GaussianPrior([0.0], PDMat([σ₀²;;]))
    dyn_ssm = LinearGaussianLatentDynamics(
        [a;;], FixedParametric((θ, _) -> [θ[1]]), PDMat([q²;;])
    )
    obs_ssm = LinearGaussianObservationProcess([1.0;;], [0.0], PDMat([r²;;]))
    model = SSMProblems.StateSpaceModel(prior_ssm, dyn_ssm, obs_ssm)

    # Sample observations and a trajectory using a fixed instance at θ = [1.0]
    fixed_model = GeneralisedFilters.fix(model, [1.0])
    _, _, ys = SSMProblems.sample(rng, fixed_model, T_len)
    x0, xs, _ = SSMProblems.sample(rng, fixed_model, T_len)
    trajectory = ReferenceTrajectory(x0, xs)

    prior = MvNormal([0.0], [4.0;;])
    ld = TrajectoryParameterLogDensity(prior, model, ys, trajectory)

    θ_test = [0.5]
    ll = LogDensityProblems.logdensity(ld, θ_test)

    # Manual reference via fix
    ll_expected =
        logpdf(prior, θ_test) +
        trajectory_logdensity(GeneralisedFilters.fix(model, θ_test), trajectory, ys)

    @test ll ≈ ll_expected
    @test LogDensityProblems.dimension(ld) == 1
end

@testitem "TrajectoryParameterLogDensity: HierarchicalSSM" begin
    using GeneralisedFilters
    using GeneralisedFilters: TimeVaryingParametric, ReferenceTrajectory
    using SSMProblems
    using LogDensityProblems
    using StableRNGs
    using Distributions
    using PDMats
    using LinearAlgebra

    rng = StableRNG(1234)

    D_outer, D_inner, D_obs = 1, 1, 1
    T_len = 5

    _, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, D_outer, D_inner, D_obs; static_arrays=false
    )

    # Promote the inner drift to TimeVaryingParametric so θ controls b
    C_in = hier_model.inner_model.dyn.b.f.C
    parametric_inner_dyn = LinearGaussianLatentDynamics(
        hier_model.inner_model.dyn.A,
        TimeVaryingParametric((θ, t, c) -> θ + C_in * c.prev_outer),
        hier_model.inner_model.dyn.Q,
    )
    parametric_hier = HierarchicalSSM(
        hier_model.outer_prior,
        hier_model.outer_dyn,
        hier_model.inner_model.prior,
        parametric_inner_dyn,
        hier_model.inner_model.obs,
    )

    x0, _, xs, _, ys = SSMProblems.sample(rng, hier_model, T_len)
    outer_traj = ReferenceTrajectory(x0, xs)

    prior = MvNormal(zeros(D_inner), 4.0 * I)
    ld = TrajectoryParameterLogDensity(prior, parametric_hier, KF(), ys, outer_traj)

    θ_test = [0.5]
    ll = LogDensityProblems.logdensity(ld, θ_test)

    # Manual reference using the same θ-positional path
    ll_expected =
        logpdf(prior, θ_test) +
        trajectory_logdensity(parametric_hier, KF(), outer_traj, ys, θ_test)

    @test ll ≈ ll_expected
    @test LogDensityProblems.dimension(ld) == 1
end
