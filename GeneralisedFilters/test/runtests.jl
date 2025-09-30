using Test
using TestItems
using TestItemRunner

@run_package_tests filter = ti -> !(:gpu in ti.tags)

include("Aqua.jl")
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
        _, _, ys = sample(rng, model, 1)

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

@testitem "Kalman filter StaticArrays" begin
    using GeneralisedFilters
    using SSMProblems
    using StableRNGs
    using StaticArrays

    D = 2
    rng = StableRNG(1234)

    μ0 = @SVector rand(rng, D)
    Σ0 = @SMatrix rand(rng, D, D)
    Σ0 = Σ0 * Σ0'
    A = @SMatrix rand(rng, D, D)
    b = @SVector rand(rng, D)
    Q = @SMatrix rand(rng, D, D)
    Q = Q * Q'
    H = @SMatrix rand(rng, D, D)
    c = @SVector rand(rng, D)
    R = @SMatrix rand(rng, D, D)
    R = R * R'

    model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)

    _, _, ys = sample(rng, model, 2)

    state, _ = GeneralisedFilters.filter(rng, model, KalmanFilter(), ys)

    # Verify returned values are still StaticArrays
    # @test ys[2] isa SVector{D,Float64}  # TODO: this fails due to use of MvNormal
    @test state.μ isa SVector{D,Float64}
    @test state.Σ isa SMatrix{D,D,Float64}
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
        _, _, ys = sample(rng, model, 2)

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
    using LogExpFunctions

    rng = StableRNG(1234)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(
        rng, 1, 1; static_arrays=true
    )
    _, _, ys = sample(rng, model, 4)

    resampler = GeneralisedFilters.GFTest.AlternatingResampler()
    bf = BF(10^6; resampler=resampler)
    bf_state, llbf = GeneralisedFilters.filter(rng, model, bf, ys)
    kf_state, llkf = GeneralisedFilters.filter(rng, model, KF(), ys)

    xs = getfield.(bf_state.particles, :state)
    log_ws = getfield.(bf_state.particles, :log_w)
    ws = softmax(log_ws)

    # Compare log-likelihood and states
    @test first(kf_state.μ) ≈ sum(first.(xs) .* ws) rtol = 1e-3
    @test llkf ≈ llbf atol = 1e-3
end

@testitem "Guided filter test" begin
    using SSMProblems
    using LogExpFunctions
    using StableRNGs

    rng = StableRNG(1234)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(
        rng, 1, 1; static_arrays=true
    )
    _, _, ys = sample(rng, model, 4)

    prop = GeneralisedFilters.GFTest.OptimalProposal(model.dyn, model.obs)
    resampler = GeneralisedFilters.GFTest.AlternatingResampler()
    gf = ParticleFilter(10^6, prop; resampler=resampler)
    gf_state, llgf = GeneralisedFilters.filter(rng, model, gf, ys)
    kf_state, llkf = GeneralisedFilters.filter(rng, model, KF(), ys)

    xs = getfield.(gf_state.particles, :state)
    log_ws = getfield.(gf_state.particles, :log_w)
    ws = softmax(log_ws)

    @test first(kf_state.μ) ≈ sum(first.(xs) .* ws) rtol = 1e-3
    @test llkf ≈ llgf atol = 1e-3
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

@testitem "Rao-Blackwellised BF test" begin
    using Distributions
    using GeneralisedFilters
    using LinearAlgebra
    using LogExpFunctions
    using SSMProblems
    using StableRNGs

    rng = StableRNG(1234)

    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, 1, 1, 1; static_arrays=true
    )
    _, _, ys = sample(rng, hier_model, 4)

    resampler = GeneralisedFilters.GFTest.AlternatingResampler()
    bf = BF(10^6; resampler=resampler)
    rbbf = RBPF(bf, KalmanFilter())

    rbbf_state, llrbbf = GeneralisedFilters.filter(rng, hier_model, rbbf, ys)
    xs = getfield.(rbbf_state.particles, :x)
    zs = getfield.(rbbf_state.particles, :z)
    log_ws = getfield.(rbbf_state.particles, :log_w)
    ws = softmax(log_ws)

    kf_state, llkf = GeneralisedFilters.filter(rng, full_model, KF(), ys)

    @test first(kf_state.μ) ≈ sum(only.(xs) .* ws) rtol = 1e-3
    @test last(kf_state.μ) ≈ sum(only.(getfield.(zs, :μ)) .* ws) rtol = 1e-3
    @test llkf ≈ llrbbf atol = 1e-3
end

@testitem "Rao-Blackwellised GF test" begin
    using LogExpFunctions
    using SSMProblems
    using StableRNGs

    rng = StableRNG(1234)

    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, 1, 1, 1; static_arrays=true
    )
    _, _, ys = sample(rng, hier_model, 4)

    prop = GeneralisedFilters.GFTest.OverdispersedProposal(dyn(hier_model).outer_dyn, 1.5)
    resampler = GeneralisedFilters.GFTest.AlternatingResampler()
    gf = ParticleFilter(10^6, prop; resampler=resampler)
    rbgf = RBPF(gf, KalmanFilter())
    rbgf_state, llrbgf = GeneralisedFilters.filter(rng, hier_model, rbgf, ys)
    xs = getfield.(rbgf_state.particles, :x)
    zs = getfield.(rbgf_state.particles, :z)
    log_ws = getfield.(rbgf_state.particles, :log_w)
    ws = softmax(log_ws)

    kf_state, llkf = GeneralisedFilters.filter(rng, full_model, KF(), ys)

    @test first(kf_state.μ) ≈ sum(only.(xs) .* ws) rtol = 1e-3
    @test last(kf_state.μ) ≈ sum(only.(getfield.(zs, :μ)) .* ws) rtol = 1e-3
    @test llkf ≈ llrbgf atol = 1e-3
end

@testitem "ABF test" begin
    using Distributions
    using GeneralisedFilters
    using LinearAlgebra
    using LogExpFunctions
    using SSMProblems
    using StableRNGs

    rng = StableRNG(1234)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(
        rng, 1, 1; static_arrays=true
    )
    _, _, ys = sample(rng, model, 4)

    bf = BF(10^6; threshold=1.0)  # APF needs resampling every step
    abf = AuxiliaryParticleFilter(bf, MeanPredictive())
    abf_state, llabf = GeneralisedFilters.filter(rng, model, abf, ys)
    kf_state, llkf = GeneralisedFilters.filter(rng, model, KF(), ys)

    xs = getfield.(abf_state.particles, :state)
    log_ws = getfield.(abf_state.particles, :log_w)
    ws = softmax(log_ws)

    @test first(kf_state.μ) ≈ sum(first.(xs) .* ws) rtol = 1e-3
    @test llkf ≈ llabf atol = 1e-3
end

@testitem "ARBF test" begin
    using Distributions
    using GeneralisedFilters
    using LinearAlgebra
    using LogExpFunctions
    using SSMProblems
    using StableRNGs

    rng = StableRNG(1234)

    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, 1, 1, 1; static_arrays=true
    )
    _, _, ys = sample(rng, hier_model, 4)

    bf = BF(10^6; threshold=1.0)  # APF needs resampling every step
    rbbf = RBPF(bf, KalmanFilter())
    arbf = AuxiliaryParticleFilter(rbbf, MeanPredictive())
    arbf_state, llarbf = GeneralisedFilters.filter(rng, hier_model, arbf, ys)
    xs = getfield.(arbf_state.particles, :x)
    zs = getfield.(arbf_state.particles, :z)
    log_ws = getfield.(arbf_state.particles, :log_w)
    ws = softmax(log_ws)

    kf_state, llkf = GeneralisedFilters.filter(rng, full_model, KF(), ys)

    @test first(kf_state.μ) ≈ sum(only.(xs) .* ws) rtol = 1e-3
    @test last(kf_state.μ) ≈ sum(only.(getfield.(zs, :μ)) .* ws) rtol = 1e-3
    @test llkf ≈ llarbf atol = 1e-3
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
    _, _, ys = sample(rng, full_model, T)

    cb = GeneralisedFilters.AncestorCallback(nothing)
    rbpf = RBPF(KalmanFilter(), N_particles)
    GeneralisedFilters.filter(rng, hier_model, rbpf, ys; callback=cb)

    # TODO: add proper test comparing to dense storage
    tree = cb.tree
    paths = GeneralisedFilters.get_ancestry(tree)
end

@testitem "BF on hierarchical model test" begin
    using LogExpFunctions
    using SSMProblems
    using StableRNGs

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
    _, _, ys = sample(rng, full_model, T)

    # Ground truth Kalman filtering
    kf_states, kf_ll = GeneralisedFilters.filter(rng, full_model, KalmanFilter(), ys)

    # Rao-Blackwellised particle filtering
    bf = BF(N_particles; threshold=0.8)
    states, ll = GeneralisedFilters.filter(rng, hier_model, bf, ys)

    # Extract final filtered states
    xs = map(p -> getproperty(p.state, :x), states.particles)
    zs = map(p -> getproperty(p.state, :z), states.particles)
    log_ws = getfield.(states.particles, :log_w)
    ws = softmax(log_ws)

    @test kf_ll ≈ ll rtol = 1e-2

    # Higher tolerance for outer state since variance is higher
    @test first(kf_states.μ) ≈ sum(only.(xs) .* ws) rtol = 1e-1
    @test last(kf_states.μ) ≈ sum(only.(zs) .* ws) rtol = 1e-1
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
    _, _, ys = sample(rng, model, K)

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
    _, _, ys = sample(rng, model, K)

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
        log_ws = getfield.(bf_state.particles, :log_w)
        ws = softmax(log_ws)
        sampled_idx = sample(rng, 1:length(bf_state), Weights(ws))
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
    using StatsBase: sample, Weights
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
    _, _, ys = sample(rng, full_model, K)

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
        log_ws = getfield.(bf_state.particles, :log_w)
        ws = softmax(log_ws)
        sampled_idx = sample(rng, 1:length(bf_state), Weights(ws))

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
