"""Unit tests for conditional SMC (particle Gibbs) and backward simulation algorithms."""

## Standard CSMC ############################################################################

@testitem "CSMC" begin
    using GeneralisedFilters
    using StableRNGs
    using LogExpFunctions: logsumexp
    using StatsBase: sample

    SEED = 1234
    Dx = 1
    Dy = 1
    K = 10
    t_smooth = 2
    N_particles = 10
    N_burnin = 1000
    N_sample = 100000

    rng = StableRNG(SEED)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)
    _, _, ys = sample(rng, model, K)

    # Kalman smoother ground truth
    state, ks_ll = GeneralisedFilters.smooth(
        rng, model, KalmanSmoother(), ys; t_smooth=t_smooth
    )

    csmc = CSMC(BF(N_particles; threshold=0.6))
    trajectory_samples = []
    lls = Float64[]

    let ref_traj = nothing
        for i in 1:(N_burnin + N_sample)
            traj, ll = GeneralisedFilters._csmc_sample(rng, model, csmc, ys, ref_traj)
            ref_traj = traj
            if i > N_burnin
                push!(trajectory_samples, traj)
                push!(lls, ll)
            end
        end
    end

    # 1/Ẑ is an unbiased estimate of 1/Z (Elements of SMC, Section 5.2)
    log_recip_likelihood_estimate = logsumexp(-lls) - log(length(lls))

    csmc_mean = sum(getindex.(trajectory_samples, t_smooth)) / N_sample
    @test csmc_mean ≈ state.μ rtol = 1e-3
    @test log_recip_likelihood_estimate ≈ -ks_ll rtol = 1e-3
end

## Rao-Blackwellised CSMC ###################################################################

@testitem "RBCSMC" begin
    using GeneralisedFilters
    using StableRNGs
    using StaticArrays
    using Statistics
    using StatsBase: sample

    SEED = 1234
    D_outer = 1
    D_inner = 1
    D_obs = 1
    K = 5
    t_smooth = 2
    T = Float64
    N_particles = 10
    N_burnin = 1000
    N_sample = 10000

    rng = StableRNG(SEED)
    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, D_outer, D_inner, D_obs, T; static_arrays=true
    )
    _, _, ys = sample(rng, full_model, K)
    ys = [SVector{1,T}(y) for y in ys]

    # Kalman smoother ground truth
    state, _ = GeneralisedFilters.smooth(
        rng, full_model, KalmanSmoother(), ys; t_smooth=t_smooth
    )

    csmc = CSMC(RBPF(BF(N_particles; threshold=0.8), KalmanFilter()))
    trajectory_samples = []

    let ref_traj = nothing
        for i in 1:(N_burnin + N_sample)
            traj, _ = GeneralisedFilters._csmc_sample(rng, hier_model, csmc, ys, ref_traj)
            ref_traj = traj
            if i > N_burnin
                push!(trajectory_samples, deepcopy(traj))
            end
        end
    end

    # Extract outer trajectories at t_smooth
    x_trajectories = getproperty.(getindex.(trajectory_samples, t_smooth), :x)

    # Smooth the inner (z) component using backward_smooth
    inner_dyn = hier_model.inner_model.dyn
    z_smoothed_means = Vector{T}(undef, N_sample)
    for i in 1:N_sample
        smoothed_z = trajectory_samples[i][K].z
        for t in (K - 1):-1:t_smooth
            filtered_z = trajectory_samples[i][t].z
            smoothed_z = backward_smooth(
                inner_dyn,
                KF(),
                t,
                filtered_z,
                smoothed_z;
                prev_outer=trajectory_samples[i][t].x,
            )
        end
        z_smoothed_means[i] = only(smoothed_z.μ)
    end

    @test state.μ[1] ≈ only(mean(x_trajectories)) rtol = 1e-2
    @test state.μ[2] ≈ mean(z_smoothed_means) rtol = 1e-3
end

## RBCSMC with Ancestor Sampling ############################################################

@testitem "RBCSMC-AS" begin
    using GeneralisedFilters
    using StableRNGs
    using Statistics
    using StatsBase: sample

    SEED = 1234
    D_outer = 1
    D_inner = 1
    D_obs = 1
    K = 5
    t_smooth = 2
    T = Float64
    N_particles = 10
    N_burnin = 200
    N_sample = 10000

    rng = StableRNG(SEED)
    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, D_outer, D_inner, D_obs, T; static_arrays=false
    )
    _, _, ys = sample(rng, full_model, K)

    # Kalman smoother ground truth
    state, _ = GeneralisedFilters.smooth(
        rng, full_model, KalmanSmoother(), ys; t_smooth=t_smooth
    )

    csmc = CSMCAS(RBPF(BF(N_particles; threshold=0.8), KalmanFilter()))
    trajectory_samples = []

    let ref_traj = nothing
        for i in 1:(N_burnin + N_sample)
            traj, _ = GeneralisedFilters._csmc_sample(rng, hier_model, csmc, ys, ref_traj)
            ref_traj = traj
            if i > N_burnin
                push!(trajectory_samples, deepcopy(traj))
            end
        end
    end

    # Extract outer trajectories at t_smooth
    x_trajectories = getproperty.(getindex.(trajectory_samples, t_smooth), :x)

    # Smooth the inner (z) component using backward_smooth
    inner_dyn = hier_model.inner_model.dyn
    z_smoothed_means = Vector{T}(undef, N_sample)
    for i in 1:N_sample
        smoothed_z = trajectory_samples[i][K].z
        for t in (K - 1):-1:t_smooth
            filtered_z = trajectory_samples[i][t].z
            smoothed_z = backward_smooth(
                inner_dyn,
                KF(),
                t,
                filtered_z,
                smoothed_z;
                prev_outer=trajectory_samples[i][t].x,
            )
        end
        z_smoothed_means[i] = only(smoothed_z.μ)
    end

    @test state.μ[1] ≈ only(mean(x_trajectories)) rtol = 1e-2
    @test state.μ[2] ≈ mean(z_smoothed_means) rtol = 1e-3
end

@testitem "Discrete RBCSMC-AS" begin
    using GeneralisedFilters
    using StableRNGs
    using StatsBase: sample
    using Statistics

    SEED = 1234
    K_outer = 3
    K_inner = 4
    T = 5
    t_smooth = 2
    N_particles = 10
    N_burnin = 200
    N_sample = 5000

    rng = StableRNG(SEED)
    joint_model, hier_model = GeneralisedFilters.GFTest.create_dummy_discrete_model(
        rng, K_outer, K_inner; obs_separation=3.0, obs_noise=0.3
    )
    _, _, _, _, ys = sample(rng, hier_model, T)

    # Ground truth: smoothed distribution from joint model
    joint_smoothed, _ = smooth(rng, joint_model, DiscreteSmoother(), ys; t_smooth=t_smooth)

    true_outer_marginal = zeros(K_outer)
    true_inner_marginal = zeros(K_inner)
    for i in 1:K_outer
        for k in 1:K_inner
            idx = (i - 1) * K_inner + k
            true_outer_marginal[i] += joint_smoothed[idx]
            true_inner_marginal[k] += joint_smoothed[idx]
        end
    end

    csmc = CSMCAS(RBPF(BF(N_particles; threshold=0.8), DiscreteFilter()))
    trajectory_samples = []

    let ref_traj = nothing
        for i in 1:(N_burnin + N_sample)
            traj, _ = GeneralisedFilters._csmc_sample(rng, hier_model, csmc, ys, ref_traj)
            ref_traj = traj
            if i > N_burnin
                push!(trajectory_samples, deepcopy(traj))
            end
        end
    end

    # Compute smoothed marginals from CSMC samples
    csmc_outer_marginal = zeros(K_outer)
    csmc_inner_marginal = zeros(K_inner)

    for traj in trajectory_samples
        rb_state = traj[t_smooth]
        csmc_outer_marginal[rb_state.x] += 1.0

        # The inner state z is a filtered distribution, need to smooth it
        smoothed_z = let s = traj[T].z
            for t in (T - 1):-1:t_smooth
                filtered_z = traj[t].z
                pred_z = predict(
                    rng,
                    hier_model.inner_model.dyn,
                    DiscreteFilter(),
                    t + 1,
                    filtered_z,
                    nothing,
                )
                s = backward_smooth(
                    hier_model.inner_model.dyn,
                    DiscreteFilter(),
                    t,
                    filtered_z,
                    s;
                    predicted=pred_z,
                )
            end
            s
        end
        csmc_inner_marginal .+= smoothed_z
    end

    csmc_outer_marginal ./= N_sample
    csmc_inner_marginal ./= N_sample

    @test csmc_outer_marginal ≈ true_outer_marginal rtol = 0.05
    @test csmc_inner_marginal ≈ true_inner_marginal rtol = 0.05
end

## CSMC AbstractMCMC Interface ##############################################################

@testitem "CSMC AbstractMCMC interface" begin
    using GeneralisedFilters
    using AbstractMCMC: AbstractMCMC
    using StableRNGs
    using StatsBase: sample

    rng = StableRNG(1234)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, 1, 1)
    _, _, ys = sample(rng, model, 5)

    csmc_model = CSMCModel(model, ys)
    csmc = CSMC(BF(10))

    # Initial step (unconditional)
    transition, state = AbstractMCMC.step(rng, csmc_model, csmc)
    @test transition isa CSMCState
    @test state isa CSMCState
    @test length(state.trajectory) == 6  # T+1 elements (indices 0:5)

    # Subsequent step (conditional)
    transition2, state2 = AbstractMCMC.step(rng, csmc_model, csmc, state)
    @test state2 isa CSMCState
    @test length(state2.trajectory) == 6

    # CSMC-BS
    csmc_bs = CSMCBS(BF(10))
    _, bs_state = AbstractMCMC.step(rng, csmc_model, csmc_bs)
    @test bs_state isa CSMCState
    @test length(bs_state.trajectory) == 6
    _, bs_state2 = AbstractMCMC.step(rng, csmc_model, csmc_bs, bs_state)
    @test bs_state2 isa CSMCState

    # Works with RBPF too
    _, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, 1, 1, 1; static_arrays=false
    )
    _, _, _, _, rb_ys = sample(rng, hier_model, 5)

    rb_model = CSMCModel(hier_model, rb_ys)
    rb_csmc = CSMCAS(RBPF(BF(10; threshold=0.8), KalmanFilter()))

    _, rb_state = AbstractMCMC.step(rng, rb_model, rb_csmc)
    @test rb_state isa CSMCState
    _, rb_state2 = AbstractMCMC.step(rng, rb_model, rb_csmc, rb_state)
    @test rb_state2 isa CSMCState
end

## Backward Simulation ######################################################################

@testitem "CSMC-BS" begin
    using GeneralisedFilters
    using StableRNGs
    using StatsBase: sample
    using Statistics

    SEED = 1234
    Dx = 1
    Dy = 1
    K = 5
    t_smooth = 3
    N_particles = 50
    N_trajectories = 1000

    rng = StableRNG(SEED)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)
    _, _, ys = sample(rng, model, K)

    # Kalman smoother ground truth
    ks_state, _ = GeneralisedFilters.smooth(
        rng, model, KalmanSmoother(), ys; t_smooth=t_smooth
    )

    csmc = CSMCBS(BF(N_particles))
    trajectory_samples = []

    let ref_traj = nothing
        for i in 1:N_trajectories
            traj, _ = GeneralisedFilters._csmc_sample(rng, model, csmc, ys, ref_traj)
            ref_traj = traj
            push!(trajectory_samples, traj)
        end
    end

    bs_mean = mean(first.(getindex.(trajectory_samples, t_smooth)))
    @test bs_mean ≈ only(ks_state.μ) rtol = 5e-2
end

@testitem "RBCSMC-BS" begin
    using GeneralisedFilters
    using StableRNGs
    using StatsBase: sample
    using Statistics

    SEED = 1234
    D_outer = 1
    D_inner = 1
    D_obs = 1
    K = 5
    t_smooth = 2
    T = Float64
    N_particles = 50
    N_burnin = 200
    N_sample = 1000

    rng = StableRNG(SEED)
    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, D_outer, D_inner, D_obs, T; static_arrays=false
    )
    _, _, ys = sample(rng, full_model, K)

    # Kalman smoother ground truth
    ks_state, _ = GeneralisedFilters.smooth(
        rng, full_model, KalmanSmoother(), ys; t_smooth=t_smooth
    )

    csmc = CSMCBS(RBPF(BF(N_particles), KalmanFilter()))
    trajectory_samples = []

    let ref_traj = nothing
        for i in 1:(N_burnin + N_sample)
            traj, _ = GeneralisedFilters._csmc_sample(rng, hier_model, csmc, ys, ref_traj)
            ref_traj = traj
            if i > N_burnin
                push!(trajectory_samples, deepcopy(traj))
            end
        end
    end

    # Extract outer trajectories at t_smooth
    x_trajectories = getproperty.(getindex.(trajectory_samples, t_smooth), :x)

    # Smooth the inner (z) component using backward_smooth
    inner_dyn = hier_model.inner_model.dyn
    z_smoothed_means = Vector{T}(undef, N_sample)
    for i in 1:N_sample
        smoothed_z = trajectory_samples[i][K].z
        for t in (K - 1):-1:t_smooth
            filtered_z = trajectory_samples[i][t].z
            smoothed_z = backward_smooth(
                inner_dyn,
                KF(),
                t,
                filtered_z,
                smoothed_z;
                prev_outer=trajectory_samples[i][t].x,
            )
        end
        z_smoothed_means[i] = only(smoothed_z.μ)
    end

    @test ks_state.μ[1] ≈ only(mean(x_trajectories)) rtol = 5e-2
    @test ks_state.μ[2] ≈ mean(z_smoothed_means) rtol = 5e-2
end
