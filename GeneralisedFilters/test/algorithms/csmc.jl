"""Unit tests for conditional SMC (particle Gibbs) and backward simulation algorithms."""

## Standard CSMC ############################################################################

@testitem "CSMC" begin
    using GeneralisedFilters
    using StableRNGs
    using PDMats
    using LinearAlgebra
    using LogExpFunctions: logsumexp
    using Random: randexp
    using StatsBase: sample, weights

    using OffsetArrays

    SEED = 1234
    Dx = 1
    Dy = 1
    K = 10
    t_smooth = 2
    T = Float64
    N_particles = 10  # Use small particle number so impact of ref state is significant
    N_burnin = 1000
    N_sample = 100000

    rng = StableRNG(SEED)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)
    _, _, ys = sample(rng, model, K)

    # Kalman smoother
    state, ks_ll = GeneralisedFilters.smooth(
        rng, model, KalmanSmoother(), ys; t_smooth=t_smooth
    )

    N_steps = N_burnin + N_sample
    bf = BF(N_particles; threshold=0.6)
    trajectory_samples = []
    lls = []

    # Run CSMC chain
    let ref_traj = nothing
        for i in 1:N_steps
            cb = GeneralisedFilters.DenseAncestorCallback(nothing)
            bf_state, ll = GeneralisedFilters.filter(
                rng, model, bf, ys; ref_state=ref_traj, callback=cb
            )
            ws = weights(bf_state)
            sampled_idx = sample(rng, 1:N_particles, ws)
            ref_traj = GeneralisedFilters.get_ancestry(cb.container, sampled_idx)
            if i > N_burnin
                push!(trajectory_samples, ref_traj)
                push!(lls, ll)
            end
        end
    end

    # The CSMC estimate of the evidence Z = p(y_{1:T}) is biased but 1 / ̂Z is actually an
    # unbiased estimate of 1 / Z. See Elements of Sequential Monte Carlo (Section 5.2)
    log_recip_likelihood_estimate = logsumexp(-lls) - log(length(lls))

    csmc_mean = sum(getindex.(trajectory_samples, t_smooth)) / N_sample
    @test csmc_mean ≈ state.μ rtol = 1e-3
    @test log_recip_likelihood_estimate ≈ -ks_ll rtol = 1e-3
end

## Rao-Blackwellised CSMC ###################################################################

@testitem "RBCSMC" begin
    using GeneralisedFilters
    using StableRNGs
    using PDMats
    using LinearAlgebra
    using Random: randexp
    using StatsBase: sample, weights
    using StaticArrays
    using Statistics

    using OffsetArrays

    SEED = 1234
    D_outer = 1
    D_inner = 1
    D_obs = 1
    K = 5
    t_smooth = 2
    T = Float64
    N_particles = 10  # Use small particle number so impact of ref state is significant
    N_burnin = 1000
    N_sample = 10000

    rng = StableRNG(SEED)
    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, D_outer, D_inner, D_obs, T; static_arrays=true
    )
    _, _, ys = sample(rng, full_model, K)
    # Convert to static arrays
    ys = [SVector{1,T}(y) for y in ys]

    # Kalman smoother
    state, _ = GeneralisedFilters.smooth(
        rng, full_model, KalmanSmoother(), ys; t_smooth=t_smooth
    )

    N_steps = N_burnin + N_sample
    rbpf = RBPF(BF(N_particles; threshold=0.8), KalmanFilter())
    trajectory_samples = []

    cb = GeneralisedFilters.DenseAncestorCallback(nothing)
    let ref_traj = nothing
        for i in 1:N_steps
            bf_state, _ = GeneralisedFilters.filter(
                rng, hier_model, rbpf, ys; ref_state=ref_traj, callback=cb
            )
            ws = weights(bf_state)
            sampled_idx = sample(rng, 1:N_particles, ws)

            full_traj = GeneralisedFilters.get_ancestry(cb.container, sampled_idx)
            if i > N_burnin
                push!(trajectory_samples, deepcopy(full_traj))
            end
            # Reference trajectory should only be nonlinear state for RBPF
            ref_traj = getproperty.(full_traj, :x)
        end
    end

    # Extract inner and outer trajectories
    x_trajectories = getproperty.(getindex.(trajectory_samples, t_smooth), :x)

    # Smooth the inner (z) component using backward_smooth
    inner_dyn = hier_model.inner_model.dyn
    z_smoothed_means = Vector{T}(undef, N_sample)
    for i in 1:N_sample
        smoothed_z = trajectory_samples[i][K].z

        for t in (K - 1):-1:t_smooth
            filtered_z = trajectory_samples[i][t].z
            # Pass prev_outer to condition the inner dynamics on the outer trajectory
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

    # Compare to ground truth
    @test state.μ[1] ≈ only(mean(x_trajectories)) rtol = 1e-2
    @test state.μ[2] ≈ mean(z_smoothed_means) rtol = 1e-3
end

## RBCSMC with Ancestor Sampling ############################################################

@testitem "RBCSMC-AS" begin
    using GeneralisedFilters
    using StableRNGs
    using PDMats
    using LinearAlgebra
    using Random: randexp
    using StatsBase: sample, weights
    using StaticArrays
    using Statistics
    using LogExpFunctions

    import SSMProblems: prior, dyn, obs
    import GeneralisedFilters: resampler, resample, move, RBState, InformationLikelihood

    using OffsetArrays

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

    # Kalman smoother
    state, _ = GeneralisedFilters.smooth(
        rng, full_model, KalmanSmoother(), ys; t_smooth=t_smooth
    )

    N_steps = N_burnin + N_sample
    rbpf = RBPF(BF(N_particles; threshold=0.8), KalmanFilter())
    trajectory_samples = []

    let ref_traj = nothing,
        predictive_likelihoods = Vector{InformationLikelihood{Vector{T},PDMat{T,Matrix{T}}}}(
            undef, K
        )

        for i in 1:N_steps
            cb = GeneralisedFilters.DenseAncestorCallback(nothing)

            # Manual filtering with ancestor resampling
            bf_state = initialise(rng, prior(hier_model), rbpf; ref_state=ref_traj)

            # Post-Init callback
            cb(hier_model, rbpf, bf_state, ys, PostInit)

            for t in 1:K
                bf_state = resample(rng, resampler(rbpf), bf_state; ref_state=ref_traj)

                ancestor_idx = nothing
                if !isnothing(ref_traj)
                    ref_rb_state = RBState(ref_traj[t], predictive_likelihoods[t])
                    ancestor_weights = map(bf_state.particles) do particle
                        ancestor_weight(particle, dyn(hier_model), rbpf, t, ref_rb_state)
                    end
                    ancestor_idx = sample(
                        rng, 1:N_particles, weights(softmax(ancestor_weights))
                    )
                end

                bf_state, ll = move(
                    rng, hier_model, rbpf, t, bf_state, ys[t]; ref_state=ref_traj
                )

                # Set ancestor index
                if !isnothing(ref_traj)
                    bf_state.particles[end] = GeneralisedFilters.Particle(
                        bf_state.particles[end].state,
                        bf_state.particles[end].log_w,
                        ancestor_idx,
                    )
                end

                # Manually trigger callback
                cb(hier_model, rbpf, t, bf_state, ys[t], PostUpdate)
            end

            ws = weights(bf_state)
            sampled_idx = sample(rng, 1:N_particles, ws)

            full_traj = GeneralisedFilters.get_ancestry(cb.container, sampled_idx)
            if i > N_burnin
                push!(trajectory_samples, deepcopy(full_traj))
            end
            # Reference trajectory should only be nonlinear state for RBPF
            ref_traj = getproperty.(full_traj, :x)

            bip = BackwardInformationPredictor(; initial_jitter=1e-8)

            pred_lik = backward_initialise(rng, hier_model.inner_model.obs, bip, K, ys[K])
            predictive_likelihoods[K] = deepcopy(pred_lik)
            for t in (K - 1):-1:1
                pred_lik = backward_predict(
                    rng,
                    hier_model.inner_model.dyn,
                    bip,
                    t,
                    pred_lik;
                    prev_outer=ref_traj[t],
                    new_outer=ref_traj[t + 1],
                )
                pred_lik = backward_update(
                    hier_model.inner_model.obs, bip, t, pred_lik, ys[t]
                )
                predictive_likelihoods[t] = deepcopy(pred_lik)
            end
        end
    end

    # Extract inner and outer trajectories
    x_trajectories = getproperty.(getindex.(trajectory_samples, t_smooth), :x)

    # Smooth the inner (z) component using backward_smooth
    inner_dyn = hier_model.inner_model.dyn
    z_smoothed_means = Vector{T}(undef, N_sample)
    for i in 1:N_sample
        smoothed_z = trajectory_samples[i][K].z

        for t in (K - 1):-1:t_smooth
            filtered_z = trajectory_samples[i][t].z
            # Pass prev_outer to condition the inner dynamics on the outer trajectory
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

    # Compare to ground truth
    @test state.μ[1] ≈ only(mean(x_trajectories)) rtol = 1e-2
    @test state.μ[2] ≈ mean(z_smoothed_means) rtol = 1e-3
end

@testitem "Discrete RBCSMC-AS" begin
    using GeneralisedFilters
    using StableRNGs
    using StatsBase: sample, weights
    using Statistics
    using LogExpFunctions

    import SSMProblems: prior, dyn, obs
    import GeneralisedFilters: resampler, resample, move, RBState, DiscreteLikelihood

    using OffsetArrays

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

    # Extract marginals from joint smoothed distribution
    true_outer_marginal = zeros(K_outer)
    true_inner_marginal = zeros(K_inner)
    for i in 1:K_outer
        for k in 1:K_inner
            idx = (i - 1) * K_inner + k
            true_outer_marginal[i] += joint_smoothed[idx]
            true_inner_marginal[k] += joint_smoothed[idx]
        end
    end

    N_steps = N_burnin + N_sample
    rbpf = RBPF(BF(N_particles; threshold=0.8), DiscreteFilter())
    trajectory_samples = []

    let ref_traj = nothing,
        predictive_likelihoods = Vector{DiscreteLikelihood{Vector{Float64}}}(undef, T)

        for i in 1:N_steps
            cb = GeneralisedFilters.DenseAncestorCallback(nothing)

            bf_state = initialise(rng, prior(hier_model), rbpf; ref_state=ref_traj)
            cb(hier_model, rbpf, bf_state, ys, PostInit)

            for t in 1:T
                bf_state = resample(rng, resampler(rbpf), bf_state; ref_state=ref_traj)

                ancestor_idx = nothing
                if !isnothing(ref_traj)
                    ref_rb_state = RBState(ref_traj[t], predictive_likelihoods[t])
                    ancestor_weights = map(bf_state.particles) do particle
                        ancestor_weight(particle, dyn(hier_model), rbpf, t, ref_rb_state)
                    end
                    ancestor_idx = sample(
                        rng, 1:N_particles, weights(softmax(ancestor_weights))
                    )
                end

                bf_state, _ = move(
                    rng, hier_model, rbpf, t, bf_state, ys[t]; ref_state=ref_traj
                )

                if !isnothing(ref_traj)
                    bf_state.particles[end] = GeneralisedFilters.Particle(
                        bf_state.particles[end].state,
                        bf_state.particles[end].log_w,
                        ancestor_idx,
                    )
                end

                cb(hier_model, rbpf, t, bf_state, ys[t], PostUpdate)
            end

            ws = weights(bf_state)
            sampled_idx = sample(rng, 1:N_particles, ws)

            full_traj = GeneralisedFilters.get_ancestry(cb.container, sampled_idx)
            if i > N_burnin
                push!(trajectory_samples, deepcopy(full_traj))
            end
            ref_traj = getproperty.(full_traj, :x)

            # Compute backward predictive likelihoods for next iteration
            bdp = BackwardDiscretePredictor()
            pred_lik = GeneralisedFilters.backward_initialise(
                rng, hier_model.inner_model.obs, bdp, T, ys[T]; num_states=K_inner
            )
            predictive_likelihoods[T] = deepcopy(pred_lik)
            for t in (T - 1):-1:1
                pred_lik = GeneralisedFilters.backward_predict(
                    rng, hier_model.inner_model.dyn, bdp, t, pred_lik
                )
                pred_lik = GeneralisedFilters.backward_update(
                    hier_model.inner_model.obs, bdp, t, pred_lik, ys[t]
                )
                predictive_likelihoods[t] = deepcopy(pred_lik)
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

## Backward Simulation ######################################################################

@testitem "Backward simulation" begin
    using GeneralisedFilters
    using StableRNGs
    using PDMats
    using LinearAlgebra
    using StatsBase: sample, weights
    using Statistics
    using LogExpFunctions

    import SSMProblems: dyn, obs, prior
    import GeneralisedFilters: resample, resampler, move, Particle

    SEED = 1234
    Dx = 1
    Dy = 1
    K = 5
    t_smooth = 3
    T = Float64
    N_particles = 50
    N_trajectories = 1000

    rng = StableRNG(SEED)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)
    _, _, ys = sample(rng, model, K)

    # Kalman smoother ground truth
    ks_state, _ = GeneralisedFilters.smooth(
        rng, model, KalmanSmoother(), ys; t_smooth=t_smooth
    )

    # Run forward filter manually and store particle states at each time step
    bf = BF(N_particles)

    # Storage for particle states at time steps 1:K
    particle_states = Vector{Vector{Particle{Vector{T},T,Int}}}(undef, K)

    # Forward filtering pass
    final_state = let state = initialise(rng, prior(model), bf)
        for t in 1:K
            state = resample(rng, resampler(bf), state)
            state, _ = move(rng, model, bf, t, state, ys[t])
            particle_states[t] = deepcopy(collect(state.particles))
        end
        state
    end

    # Backward simulation: sample M trajectories
    trajectory_samples = Vector{Vector{T}}(undef, N_trajectories)

    for m in 1:N_trajectories
        # Sample from final distribution
        final_ws = weights(final_state)
        idx = sample(rng, 1:N_particles, final_ws)

        # Initialize trajectory with sampled final state
        traj = Vector{Vector{T}}(undef, K)
        traj[K] = particle_states[K][idx].state

        # Backward simulation pass - resample ancestors using backward weights
        for t in (K - 1):-1:1
            particles_t = particle_states[t]

            # Compute backward weights: w_t^i * f(x_{t+1} | x_t^i)
            ref_state = traj[t + 1]
            backward_ws = map(particles_t) do particle
                ancestor_weight(particle, dyn(model), bf, t + 1, ref_state)
            end

            # Sample new ancestor
            idx = sample(rng, 1:N_particles, weights(softmax(backward_ws)))
            traj[t] = particles_t[idx].state
        end

        trajectory_samples[m] = [traj[t][1] for t in 1:K]
    end

    # Extract samples at t_smooth and compare to Kalman smoother
    bs_mean = mean(getindex.(trajectory_samples, t_smooth))
    @test bs_mean ≈ only(ks_state.μ) rtol = 5e-2
end

@testitem "RB backward simulation" begin
    using GeneralisedFilters
    using StableRNGs
    using PDMats
    using LinearAlgebra
    using StatsBase: sample, weights
    using Statistics
    using LogExpFunctions
    using Distributions: MvNormal

    import SSMProblems: dyn, obs, prior
    import GeneralisedFilters:
        RBState, InformationLikelihood, resample, resampler, move, Particle

    SEED = 1234
    D_outer = 1
    D_inner = 1
    D_obs = 1
    K = 5
    t_smooth = 2
    T = Float64
    N_particles = 50
    N_trajectories = 1000

    rng = StableRNG(SEED)
    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, D_outer, D_inner, D_obs, T; static_arrays=false
    )
    _, _, ys = sample(rng, full_model, K)

    # Kalman smoother ground truth on full model
    ks_state, _ = GeneralisedFilters.smooth(
        rng, full_model, KalmanSmoother(), ys; t_smooth=t_smooth
    )

    # Run RBPF forward filter manually and store particle states
    rbpf = RBPF(BF(N_particles), KalmanFilter())

    # Initialize and run first step to get concrete types
    init_state = initialise(rng, prior(hier_model), rbpf)
    init_state = resample(rng, resampler(rbpf), init_state)
    init_state, _ = move(rng, hier_model, rbpf, 1, init_state, ys[1])

    # Storage for particle states at time steps 1:K
    particle_states = Vector{typeof(collect(init_state.particles))}(undef, K)
    particle_states[1] = deepcopy(collect(init_state.particles))

    # Forward filtering pass for remaining steps
    final_state = let state = init_state
        for t in 2:K
            state = resample(rng, resampler(rbpf), state)
            state, _ = move(rng, hier_model, rbpf, t, state, ys[t])
            particle_states[t] = deepcopy(collect(state.particles))
        end
        state
    end

    # Backward simulation: sample M trajectories
    x_samples = Vector{T}(undef, N_trajectories)
    z_samples = Vector{T}(undef, N_trajectories)

    for m in 1:N_trajectories
        # Sample from final distribution
        final_ws = weights(final_state)
        idx = sample(rng, 1:N_particles, final_ws)

        # Initialize trajectory with sampled final state
        traj = Vector{eltype(particle_states[1]).parameters[1]}(undef, K)
        traj[K] = particle_states[K][idx].state

        # Extract outer trajectory for computing backward likelihoods
        outer_traj = Vector{Vector{T}}(undef, K)
        outer_traj[K] = traj[K].x

        # Compute backward predictive likelihoods for this trajectory
        bip = BackwardInformationPredictor(; initial_jitter=1e-8)
        pred_lik = backward_initialise(rng, hier_model.inner_model.obs, bip, K, ys[K])
        predictive_likelihoods = Vector{typeof(pred_lik)}(undef, K)
        predictive_likelihoods[K] = deepcopy(pred_lik)

        # Backward simulation pass
        for t in (K - 1):-1:1
            particles_t = particle_states[t]

            # Build reference state with backward predictive likelihood
            ref_rb_state = RBState(outer_traj[t + 1], predictive_likelihoods[t + 1])

            # Compute backward weights using ancestor_weight
            backward_ws = map(particles_t) do particle
                ancestor_weight(particle, dyn(hier_model), rbpf, t, ref_rb_state)
            end

            # Sample new ancestor
            new_idx = sample(rng, 1:N_particles, weights(softmax(backward_ws)))
            traj[t] = particles_t[new_idx].state
            outer_traj[t] = traj[t].x

            # Compute backward predictive likelihood at time t
            pred_lik = backward_predict(
                rng,
                hier_model.inner_model.dyn,
                bip,
                t,
                predictive_likelihoods[t + 1];
                prev_outer=outer_traj[t],
                new_outer=outer_traj[t + 1],
            )
            pred_lik = backward_update(hier_model.inner_model.obs, bip, t, pred_lik, ys[t])
            predictive_likelihoods[t] = deepcopy(pred_lik)
        end

        # Store outer state sample at t_smooth
        x_samples[m] = only(traj[t_smooth].x)

        # Smooth the inner (z) component using backward_smooth
        inner_dyn = hier_model.inner_model.dyn
        smoothed_z = traj[K].z
        for t in (K - 1):-1:t_smooth
            filtered_z = traj[t].z
            smoothed_z = backward_smooth(
                inner_dyn, KF(), t, filtered_z, smoothed_z; prev_outer=traj[t].x
            )
        end
        z_samples[m] = only(smoothed_z.μ)
    end

    # Compare to ground truth
    @test ks_state.μ[1] ≈ mean(x_samples) rtol = 5e-2
    @test ks_state.μ[2] ≈ mean(z_samples) rtol = 5e-2
end
