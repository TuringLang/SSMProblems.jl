using Test
using TestItems
using TestItemRunner

@run_package_tests

include("Aqua.jl")
include("batch_kalman_test.jl")
include("resamplers.jl")

@testitem "Kalman filter test" begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs

    SEED = 1234
    Dx = 3
    Dys = [2, 3, 4]

    for Dy in Dys
        rng = StableRNG(1234)
        model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)
        _, ys = sample(rng, model, 1)

        filtered, ll = GeneralisedFilters.filter(rng, model, KalmanFilter(), ys)

        # Let Z = [X0, X1, Y1] be the joint state vector
        μ_Z, Σ_Z = GeneralisedFilters.GFTest._compute_joint(model, 1)

        # Condition on observations using formula for MVN conditional distribution. See: 
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        y = only(ys)
        I_x = (Dx + 1):(2Dx)
        I_y = (2Dx + 1):(2Dx + Dy)
        μ_X1 = μ_Z[I_x] + Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ (y - μ_Z[I_y]))
        Σ_X1 = Σ_Z[I_x, I_x] - Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ Σ_Z[I_y, I_x])

        @test filtered.μ ≈ μ_X1
        @test filtered.Σ ≈ Σ_X1

        # Exact marginal distribution to test log-likelihood
        μ_Y1 = μ_Z[I_y]
        Σ_Y1 = Σ_Z[I_y, I_y]
        LinearAlgebra.hermitianpart!(Σ_Y1)
        true_ll = logpdf(MvNormal(μ_Y1, Σ_Y1), y)
        @test ll ≈ true_ll
    end
end

@testitem "Kalman smoother test" begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs

    SEED = 1234
    Dx = 3
    Dys = [2, 3, 4]

    for Dy in Dys
        rng = StableRNG(1234)
        model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)
        _, ys = sample(rng, model, 2)

        states, ll = GeneralisedFilters.smooth(rng, model, KalmanSmoother(), ys)

        # Let Z = [X0, X1, X2, Y1, Y2] be the joint state vector
        μ_Z, Σ_Z = GeneralisedFilters.GFTest._compute_joint(model, 2)

        # Condition on observations using formula for MVN conditional distribution. See: 
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        y = [ys[1]; ys[2]]
        I_x = (Dx + 1):(2Dx)  # just X1
        I_y = (3Dx + 1):(3Dx + 2Dy)  # Y1 and Y2
        μ_X1 = μ_Z[I_x] + Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ (y - μ_Z[I_y]))
        Σ_X1 = Σ_Z[I_x, I_x] - Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ Σ_Z[I_y, I_x])

        @test states.μ ≈ μ_X1
        @test states.Σ ≈ Σ_X1
    end
end

@testitem "Bootstrap filter test" begin
    using SSMProblems
    using StableRNGs
    using LogExpFunctions: softmax

    rng = StableRNG(1234)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, 1, 1)
    _, ys = sample(rng, model, 10)

    bf = BF(2^12; threshold=0.8)
    bf_state, llbf = GeneralisedFilters.filter(rng, model, bf, ys)
    kf_state, llkf = GeneralisedFilters.filter(rng, model, KF(), ys)

    xs = bf_state.particles
    ws = softmax(bf_state.log_weights)

    # Compare log-likelihood and states
    @test first(kf_state.μ) ≈ sum(first.(xs) .* ws) rtol = 1e-2
    @test llkf ≈ llbf atol = 1e-2
end

@testitem "Guided filter test" begin
    using SSMProblems
    using LogExpFunctions: softmax
    using StableRNGs
    using Distributions
    using LinearAlgebra

    struct LinearGaussianProposal <: GeneralisedFilters.AbstractProposal end

    function SSMProblems.distribution(
        model::AbstractStateSpaceModel,
        kernel::LinearGaussianProposal,
        iter::Integer,
        state,
        observation;
        kwargs...,
    )
        A, b, Q = GeneralisedFilters.calc_params(model.dyn, iter; kwargs...)
        pred = GeneralisedFilters.GaussianDistribution(A * state + b, Q)
        prop, _ = GeneralisedFilters.update(model, KF(), iter, pred, observation; kwargs...)
        return MvNormal(prop.μ, hermitianpart(prop.Σ))
    end

    rng = StableRNG(1234)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, 1, 1)
    _, ys = sample(rng, model, 10)

    algo = PF(2^10, LinearGaussianProposal(); threshold=0.6)
    kf_states, kf_ll = GeneralisedFilters.filter(rng, model, KalmanFilter(), ys)
    pf_states, pf_ll = GeneralisedFilters.filter(rng, model, algo, ys)
    xs = pf_states.particles
    ws = softmax(pf_states.log_weights)

    # Compare log-likelihood and states
    @test first(kf_states.μ) ≈ sum(first.(xs) .* ws) rtol = 1e-2
    @test kf_ll ≈ pf_ll rtol = 1e-2
end

@testitem "Forward algorithm test" begin
    using GeneralisedFilters
    using Distributions
    using StableRNGs
    using SSMProblems

    rng = StableRNG(1234)
    α0 = rand(rng, 3)
    α0 = α0 / sum(α0)
    P = rand(rng, 3, 3)
    P = P ./ sum(P; dims=2)

    struct MixtureModelObservation{T<:Real,MT<:AbstractVector{T}} <: ObservationProcess
        μs::MT
    end

    function SSMProblems.logdensity(
        obs::MixtureModelObservation{T},
        step::Integer,
        state::Integer,
        observation;
        kwargs...,
    ) where {T}
        return logpdf(Normal(obs.μs[state], one(T)), observation)
    end

    μs = [0.0, 1.0, 2.0]

    prior = HomogenousDiscretePrior(α0)
    dyn = HomogeneousDiscreteLatentDynamics(P)
    obs = MixtureModelObservation(μs)
    model = StateSpaceModel(prior, dyn, obs)

    observations = [rand(rng)]

    fw = ForwardAlgorithm()
    state, ll = GeneralisedFilters.filter(model, fw, observations)

    # Brute force calculations of each conditional path probability p(x_{1:T} | y_{1:T})
    T = 1
    K = 3
    y = only(observations)
    path_probs = Dict{Tuple{Int,Int},Float64}()
    for x0 in 1:K, x1 in 1:K
        prior_prob = α0[x0] * P[x0, x1]
        likelihood = exp(SSMProblems.logdensity(obs, 1, x1, y))
        path_probs[(x0, x1)] = prior_prob * likelihood
    end
    marginal = sum(values(path_probs))

    filtered_paths = Base.filter(((k, v),) -> k[end] == 1, path_probs)
    @test state[1] ≈ sum(values(filtered_paths)) / marginal
    @test ll ≈ log(marginal)
end

@testitem "Kalman-RBPF test" begin
    using LogExpFunctions: softmax
    using StableRNGs
    using StatsBase

    SEED = 1234
    D_outer = 2
    D_inner = 3
    D_obs = 2
    T = 5
    N_particles = 10^4

    rng = StableRNG(SEED)

    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, D_outer, D_inner, D_obs
    )
    _, ys = sample(rng, full_model, T)

    # Ground truth Kalman filtering
    kf_states, kf_ll = GeneralisedFilters.filter(rng, full_model, KalmanFilter(), ys)

    # Rao-Blackwellised particle filtering
    rbpf = RBPF(KalmanFilter(), N_particles)
    states, ll = GeneralisedFilters.filter(rng, hier_model, rbpf, ys)

    # Extract final filtered states
    xs = map(p -> getproperty(p, :x), states.particles)
    zs = map(p -> getproperty(p, :z), states.particles)
    log_ws = states.log_weights

    @test kf_ll ≈ ll rtol = 1e-2

    weights = Weights(softmax(log_ws))

    # Higher tolerance for outer state since variance is higher
    @test first(kf_states.μ) ≈ sum(first.(xs) .* weights) rtol = 1e-1
    @test last(kf_states.μ) ≈ sum(last.(getproperty.(zs, :μ)) .* weights) rtol = 1e-2
end

@testitem "GPU Kalman-RBPF test" tags = [:gpu] begin
    using CUDA
    using LinearAlgebra
    using NNlib
    using SSMProblems
    using StableRNGs

    SEED = 1234
    D_outer = 2
    D_inner = 3
    D_obs = 2
    T = 5
    N_particles = 10^4
    ET = Float32

    rng = StableRNG(SEED)

    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, D_outer, D_inner, D_obs, ET
    )
    _, _, ys = sample(rng, full_model, T)

    # Ground truth Kalman filtering
    kf_state, kf_ll = GeneralisedFilters.filter(full_model, KalmanFilter(), ys)

    # Rao-Blackwellised particle filtering
    rbpf = BatchRBPF(BatchKalmanFilter(N_particles), N_particles)
    states, ll = GeneralisedFilters.filter(hier_model, rbpf, ys)

    # Extract final filtered states
    xs = states.particles.xs
    zs = states.particles.zs
    log_ws = states.log_weights

    weights = softmax(log_ws)
    reshaped_weights = reshape(weights, (1, length(weights)))

    @test kf_ll ≈ ll rtol = 1e-2
    @test first(kf_state.μ) ≈ sum(xs[1, :] .* weights) rtol = 1e-1
    @test last(kf_state.μ) ≈ sum(zs.μs[end, :] .* weights) rtol = 1e-2
    @test eltype(xs) == ET
end

@testitem "RBPF ancestory test" begin
    using SSMProblems
    using StableRNGs

    SEED = 1234
    T = 5
    N_particles = 100

    rng = StableRNG(SEED)
    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, 1, 1, 1
    )
    _, ys = sample(rng, full_model, T)

    cb = GeneralisedFilters.AncestorCallback(nothing)
    rbpf = RBPF(KalmanFilter(), N_particles)
    GeneralisedFilters.filter(rng, hier_model, rbpf, ys; callback=cb)

    # TODO: add proper test comparing to dense storage
    tree = cb.tree
    paths = GeneralisedFilters.get_ancestry(tree)
end

@testitem "BF on hierarchical model test" begin
    using LogExpFunctions: softmax
    using StableRNGs
    using StatsBase

    SEED = 1234
    D_outer = 1
    D_inner = 1
    D_obs = 1
    T = 5
    N_particles = 10^4

    rng = StableRNG(SEED)

    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, D_outer, D_inner, D_obs
    )
    _, ys = sample(rng, full_model, T)

    # Ground truth Kalman filtering
    kf_states, kf_ll = GeneralisedFilters.filter(rng, full_model, KalmanFilter(), ys)

    # Rao-Blackwellised particle filtering
    bf = BF(N_particles)
    states, ll = GeneralisedFilters.filter(rng, hier_model, bf, ys)

    # Extract final filtered states
    xs = map(p -> getproperty(p, :x), states.particles)
    zs = map(p -> getproperty(p, :z), states.particles)
    log_ws = states.log_weights

    @test kf_ll ≈ ll rtol = 1e-2

    weights = Weights(softmax(log_ws))

    # Higher tolerance for outer state since variance is higher
    @test first(kf_states.μ) ≈ sum(only.(xs) .* weights) rtol = 1e-1
    @test last(kf_states.μ) ≈ sum(only.(zs) .* weights) rtol = 1e-1
end

@testitem "Dense ancestry test" begin
    using GeneralisedFilters
    using StableRNGs
    using PDMats
    using LinearAlgebra
    using LogExpFunctions: softmax
    using Random: randexp, AbstractRNG
    using StatsBase: sample, Weights

    using OffsetArrays

    struct DummyResampler <: GeneralisedFilters.AbstractResampler end

    function GeneralisedFilters.sample_ancestors(
        rng::AbstractRNG, resampler::DummyResampler, weights::AbstractVector
    )
        return [mod1(a - 1, length(weights)) for a in 1:length(weights)]
    end

    SEED = 1234
    K = 5
    N_particles = max(10, K + 2)

    rng = StableRNG(SEED)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, 1, 1)
    _, ys = sample(rng, model, K)

    ref_traj = OffsetVector([rand(rng, 1) for _ in 0:K], -1)

    bf = BF(N_particles; threshold=1.0, resampler=DummyResampler())
    cb = GeneralisedFilters.DenseAncestorCallback(nothing)
    bf_state, _ = GeneralisedFilters.filter(
        rng, model, bf, ys; ref_state=ref_traj, callback=cb
    )

    traj = GeneralisedFilters.get_ancestry(cb.container, N_particles)
    true_traj = [cb.container.particles[t][N_particles - K + t] for t in 0:K]

    @test traj.parent == true_traj
    @test GeneralisedFilters.get_ancestry(cb.container, 1) == ref_traj
end

@testitem "CSMC test" begin
    using GeneralisedFilters
    using StableRNGs
    using PDMats
    using LinearAlgebra
    using LogExpFunctions: softmax, logsumexp
    using Random: randexp
    using StatsBase: sample, Weights

    using OffsetArrays

    SEED = 1234
    Dx = 1
    Dy = 1
    K = 10
    t_smooth = 2
    T = Float64
    N_particles = 10
    N_burnin = 1000
    N_sample = 10000

    rng = StableRNG(SEED)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)
    _, ys = sample(rng, model, K)

    # Kalman smoother
    state, ks_ll = GeneralisedFilters.smooth(
        rng, model, KalmanSmoother(), ys; t_smooth=t_smooth
    )

    N_steps = N_burnin + N_sample
    bf = BF(N_particles; threshold=0.6)
    ref_traj = nothing
    trajectory_samples = []
    lls = []

    for i in 1:N_steps
        cb = GeneralisedFilters.DenseAncestorCallback(nothing)
        bf_state, ll = GeneralisedFilters.filter(
            rng, model, bf, ys; ref_state=ref_traj, callback=cb
        )
        weights = softmax(bf_state.log_weights)
        sampled_idx = sample(rng, 1:length(weights), Weights(weights))
        global ref_traj = GeneralisedFilters.get_ancestry(cb.container, sampled_idx)
        if i > N_burnin
            push!(trajectory_samples, ref_traj)
            push!(lls, ll)
        end
    end

    # The CSMC estimate of the evidence Z = p(y_{1:T}) is biased but 1 / ̂Z is actually an
    # unbiased estimate of 1 / Z. See Elements of Sequential Monte Carlo (Section 5.2)
    log_recip_likelihood_estimate = logsumexp(-lls) - log(length(lls))

    csmc_mean = sum(getindex.(trajectory_samples, t_smooth)) / N_sample
    @test csmc_mean ≈ state.μ rtol = 1e-2
    @test log_recip_likelihood_estimate ≈ -ks_ll rtol = 1e-2
end

@testitem "RBCSMC test" begin
    using GeneralisedFilters
    using StableRNGs
    using PDMats
    using LinearAlgebra
    using LogExpFunctions: softmax
    using Random: randexp
    using StatsBase
    using StaticArrays

    using OffsetArrays

    SEED = 1234
    D_outer = 1
    D_inner = 1
    D_obs = 1
    K = 5
    t_smooth = 2
    T = Float64
    N_particles = 100
    N_burnin = 200
    N_sample = 2000

    rng = StableRNG(SEED)
    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, D_outer, D_inner, D_obs, T; static_arrays=true
    )
    _, ys = sample(rng, full_model, K)

    # Kalman smoother
    state, _ = GeneralisedFilters.smooth(
        rng, full_model, KalmanSmoother(), ys; t_smooth=t_smooth
    )

    N_steps = N_burnin + N_sample
    rbpf = RBPF(KalmanFilter(), N_particles; threshold=0.6)
    ref_traj = nothing
    trajectory_samples = []

    for i in 1:N_steps
        cb = GeneralisedFilters.DenseAncestorCallback(nothing)
        bf_state, _ = GeneralisedFilters.filter(
            rng, hier_model, rbpf, ys; ref_state=ref_traj, callback=cb
        )
        weights = softmax(bf_state.log_weights)
        sampled_idx = sample(rng, 1:length(weights), Weights(weights))

        global ref_traj = GeneralisedFilters.get_ancestry(cb.container, sampled_idx)
        if i > N_burnin
            push!(trajectory_samples, deepcopy(ref_traj))
        end
        # Reference trajectory should only be nonlinear state for RBPF
        ref_traj = getproperty.(ref_traj, :x)
    end

    # Extract inner and outer trajectories
    x_trajectories = getproperty.(getindex.(trajectory_samples, t_smooth), :x)

    # Manually perform smoothing until we have a cleaner interface
    A = hier_model.inner_model.dyn.A
    b = hier_model.inner_model.dyn.b
    C = hier_model.inner_model.dyn.C
    Q = hier_model.inner_model.dyn.Q
    z_smoothed_means = Vector{T}(undef, N_sample)
    for i in 1:N_sample
        μ = trajectory_samples[i][K].z.μ
        Σ = trajectory_samples[i][K].z.Σ

        for t in (K - 1):-1:t_smooth
            μ_filt = trajectory_samples[i][t].z.μ
            Σ_filt = trajectory_samples[i][t].z.Σ
            μ_pred = A * μ_filt + b + C * trajectory_samples[i][t].x
            Σ_pred = A * Σ_filt * A' + Q

            G = Σ_filt * A' * inv(Σ_pred)
            μ = μ_filt .+ G * (μ .- μ_pred)
            Σ = Σ_filt .+ G * (Σ .- Σ_pred) * G'
        end

        z_smoothed_means[i] = only(μ)
    end

    # Compare to ground truth
    @test state.μ[1] ≈ only(mean(x_trajectories)) rtol = 1e-2
    @test state.μ[2] ≈ mean(z_smoothed_means) rtol = 1e-3
end

@testitem "GPU Conditional Kalman-RBPF execution test" tags = [:gpu] begin
    using CUDA
    using OffsetArrays
    using SSMProblems
    using StableRNGs

    SEED = 1234
    D_outer = 2
    D_inner = 3
    D_obs = 2
    K = 5
    T = Float32
    N_particles = 1000

    rng = StableRNG(1234)

    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, D_outer, D_inner, D_obs, T
    )
    _, _, ys = sample(rng, full_model, K)

    # Generate random reference trajectory
    ref_trajectory = [CuArray(rand(rng, T, D_outer, 1)) for _ in 0:K]
    ref_trajectory = OffsetVector(ref_trajectory, -1)

    rbpf = BatchRBPF(BatchKalmanFilter(N_particles), N_particles)
    states, ll = GeneralisedFilters.filter(hier_model, rbpf, ys; ref_state=ref_trajectory)

    # Check returned type
    @test typeof(ll) == T
end

@testitem "GPU-RBPF ancestory test" tags = [:gpu] begin
    using GeneralisedFilters
    using CUDA
    using LinearAlgebra
    using SSMProblems
    using StableRNGs

    SEED = 1234
    D_outer = 2
    D_inner = 3
    D_obs = 2
    K = 5
    T = Float32
    N_particles = 10^5

    rng = StableRNG(1234)

    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, D_outer, D_inner, D_obs, T
    )
    _, _, ys = sample(rng, full_model, K)

    # Manually create tree to force expansion on second step
    M = N_particles * 2 - 1
    tree = GeneralisedFilters.ParallelParticleTree(
        GeneralisedFilters.BatchRaoBlackwellisedParticles(
            CuArray{T}(undef, D_outer, N_particles),
            GeneralisedFilters.BatchGaussianDistribution(
                CuArray{T}(undef, D_inner, N_particles),
                CuArray{T}(undef, D_inner, D_inner, N_particles),
            ),
        ),
        M,
    )

    rbpf = BatchRBPF(BatchKalmanFilter(N_particles), N_particles)
    cb = GeneralisedFilters.ParallelAncestorCallback(tree)
    states, ll = GeneralisedFilters.filter(hier_model, rbpf, ys; callback=cb)

    # TODO: add proper test comparing to dense storage
    ancestry = GeneralisedFilters.get_ancestry(tree, K)
end

@testitem "GPU Conditional Kalman-RBPF validity test" tags = [:gpu, :long] begin
    using GeneralisedFilters
    using CUDA
    using NNlib
    using OffsetArrays
    using StableRNGs
    using StatsBase

    SEED = 1234
    D_outer = 1
    D_inner = 1
    D_obs = 1
    K = 3
    t_smooth = 2
    T = Float32
    N_particles = 10000
    N_burnin = 100
    N_sample = 2000

    rng = StableRNG(1234)

    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, D_outer, D_inner, D_obs, T
    )
    _, _, ys = sample(rng, full_model, K)

    # Kalman smoother
    state, _ = GeneralisedFilters.smooth(
        rng, full_model, KalmanSmoother(), ys; t_smooth=t_smooth
    )

    particle_template = GeneralisedFilters.BatchRaoBlackwellisedParticles(
        CuArray{T}(undef, D_outer, N_particles),
        GeneralisedFilters.BatchGaussianDistribution(
            CuArray{T}(undef, D_inner, N_particles),
            CuArray{T}(undef, D_inner, D_inner, N_particles),
        ),
    )
    particle_type = typeof(particle_template)

    N_steps = N_burnin + N_sample
    M = floor(Int64, N_particles * log(N_particles))
    rbpf = BatchRBPF(BatchKalmanFilter(N_particles), N_particles; threshold=1.0)
    ref_traj = nothing
    trajectory_samples = Vector{OffsetArray{particle_type,1,Vector{particle_type}}}(
        undef, N_sample
    )

    for i in 1:N_steps
        tree = GeneralisedFilters.ParallelParticleTree(deepcopy(particle_template), M)
        cb = GeneralisedFilters.ParallelAncestorCallback(tree)
        rbpf_state, _ = GeneralisedFilters.filter(
            hier_model, rbpf, ys; ref_state=ref_traj, callback=cb
        )
        weights = softmax(rbpf_state.log_weights)
        ancestors = GeneralisedFilters.sample_ancestors(rng, Multinomial(), weights)
        sampled_idx = CUDA.@allowscalar ancestors[1]
        global ref_traj = GeneralisedFilters.get_ancestry(tree, sampled_idx, K)
        if i > N_burnin
            trajectory_samples[i - N_burnin] = ref_traj
        end
        # Reference trajectory should only be nonlinear state for RBPF
        ref_traj = getproperty.(ref_traj, :xs)
    end

    # Extract inner and outer trajectories
    x_trajectories = getproperty.(getindex.(trajectory_samples, t_smooth), :xs)
    z_trajectories = getproperty.(getindex.(trajectory_samples, t_smooth), :zs)

    # Compare to ground truth
    CUDA.@allowscalar begin
        @test state.μ[1] ≈ only(mean(x_trajectories)) rtol = 1e-1
        @test state.μ[2] ≈ only(mean(getproperty.(z_trajectories, :μs))) rtol = 1e-1
    end
end
