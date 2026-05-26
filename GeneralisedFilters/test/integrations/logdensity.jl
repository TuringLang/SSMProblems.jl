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

## SSMParameterLogDensity ######################################################################

@testitem "SSMParameterLogDensity: regular SSM" begin
    using GeneralisedFilters
    using SSMProblems
    using LogDensityProblems
    using StableRNGs
    using Distributions
    using PDMats
    using LinearAlgebra
    using GeneralisedFilters: ReferenceTrajectory

    rng = StableRNG(1234)

    # Simple 1D model: x_t = a * x_{t-1} + b + noise, y_t = x_t + noise
    # Unknown parameter: b (drift)
    a = 0.8
    q² = 0.1
    r² = 0.5
    σ₀² = 1.0
    T_len = 5

    function build_ssm(θ)
        return create_homogeneous_linear_gaussian_model(
            [0.0],
            PDMat([σ₀²;;]),
            [a;;],
            [θ[1]],
            PDMat([q²;;]),
            [1.0;;],
            [0.0],
            PDMat([r²;;]),
        )
    end

    true_b = 1.0
    true_ssm = build_ssm([true_b])
    _, _, ys = SSMProblems.sample(rng, true_ssm, T_len)

    # Sample a trajectory
    x0, xs, _ = SSMProblems.sample(rng, true_ssm, T_len)
    trajectory = ReferenceTrajectory(x0, xs)

    prior = MvNormal([0.0], [4.0;;])
    pssm = ParameterisedSSM(build_ssm, ys)
    ld = SSMParameterLogDensity(prior, pssm, trajectory)

    θ_test = [0.5]
    ll = LogDensityProblems.logdensity(ld, θ_test)

    # Manual
    model = build_ssm(θ_test)
    ll_expected = logpdf(prior, θ_test) + trajectory_logdensity(model, trajectory, ys)

    @test ll ≈ ll_expected
    @test LogDensityProblems.dimension(ld) == 1
end

@testitem "SSMParameterLogDensity: HierarchicalSSM" begin
    using GeneralisedFilters
    using SSMProblems
    using LogDensityProblems
    using StableRNGs
    using Distributions
    using PDMats
    using LinearAlgebra
    using GeneralisedFilters: ReferenceTrajectory

    rng = StableRNG(1234)

    D_outer, D_inner, D_obs = 1, 1, 1
    T_len = 5

    _, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, D_outer, D_inner, D_obs; static_arrays=false
    )
    x0, _, xs, _, ys = SSMProblems.sample(rng, hier_model, T_len)
    outer_traj = ReferenceTrajectory(x0, xs)

    # Parameterise the model by b (inner dynamics offset)
    fixed_model = hier_model
    build_hier(θ) = GeneralisedFilters.GFTest.with_inner_drift(fixed_model, θ)

    prior = MvNormal(zeros(D_inner), 4.0 * I)
    pssm = ParameterisedSSM(build_hier, ys)
    ld = SSMParameterLogDensity(prior, pssm, KF(), outer_traj)

    θ_test = [0.5]
    ll = LogDensityProblems.logdensity(ld, θ_test)

    model = build_hier(θ_test)
    ll_expected = logpdf(prior, θ_test) + trajectory_logdensity(model, KF(), outer_traj, ys)

    @test ll ≈ ll_expected
    @test LogDensityProblems.dimension(ld) == 1
end
