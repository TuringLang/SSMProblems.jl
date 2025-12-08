using Test
using TestItems
using TestItemRunner

@run_package_tests filter = ti -> !(:gpu in ti.tags)

include("Aqua.jl")
include("type_stability.jl")
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

@testitem "Backward information filter test" begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs

    SEED = 1234
    Dx = 3
    Dys = [2, 3, 4]
    T = 4

    for Dy in Dys
        rng = StableRNG(1234)
        model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)
        _, _, ys = sample(rng, model, T)

        # Perform backward information filtering
        BIF = BackwardInformationPredictor()
        predictive_likelihood = backward_initialise(rng, model.obs, BIF, T, ys[T])
        predictive_likelihood = backward_predict(
            rng, model.dyn, BIF, T - 1, predictive_likelihood
        )
        predictive_likelihood = backward_update(
            model.obs, BIF, T - 1, predictive_likelihood, ys[T - 1]
        )
        predictive_likelihood = backward_predict(
            rng, model.dyn, BIF, T - 2, predictive_likelihood
        )
        predictive_likelihood = backward_update(
            model.obs, BIF, T - 2, predictive_likelihood, ys[T - 2]
        )

        # Assuming homogenous
        A, b, Q = GeneralisedFilters.calc_params(model.dyn, T - 1)
        H, c, R = GeneralisedFilters.calc_params(model.obs, T;)
        F = [H; H * A; H * A^2]
        g = [c; H * b + c; H * (A * b + b) + c]

        #! format: off
        Σ = [
            R               zeros(Dy, Dy)    zeros(Dy, Dy);
            zeros(Dy, Dy)   H * Q * H' + R   H * Q * A' * H';
            zeros(Dy, Dy)   H * A * Q * H'   H * (A * Q * A' + Q) * H' + R
        ]
        #! format: on

        λ_true = F' * inv(Σ) * (vcat(ys[(T - 2):T]...) .- g)
        Ω_true = F' * inv(Σ) * F
        λ, Ω = GeneralisedFilters.natural_params(predictive_likelihood)

        @test λ ≈ λ_true
        @test Ω ≈ Ω_true
    end
end

@testitem "Kalman filter StaticArrays" begin
    using GeneralisedFilters
    using SSMProblems
    using StableRNGs
    using StaticArrays
    using PDMats

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

    model = create_homogeneous_linear_gaussian_model(
        μ0, PDMat(Σ0), A, b, PDMat(Q), H, c, PDMat(R)
    )

    _, _, ys = sample(rng, model, 2)

    state, _ = GeneralisedFilters.filter(rng, model, KalmanFilter(), ys)

    # Verify returned values are still StaticArrays
    @test ys[2] isa SVector{D,Float64}
    @test state.μ isa SVector{D,Float64}
    @test state.Σ isa PDMat{Float64,SMatrix{D,D,Float64,D * D}}
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
        model = GeneralisedFilters.GFTest.create_linear_gaussian_model(
            rng, Dx, Dy; static_arrays=true
        )
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
    using StatsBase: weights

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
    ws = weights(bf_state)

    # Compare log-likelihood and states
    @test first(kf_state.μ) ≈ sum(first.(xs) .* ws) rtol = 1e-3
    @test llkf ≈ llbf atol = 1e-3
end

@testitem "Guided filter test" begin
    using SSMProblems
    using StableRNGs
    using StatsBase: weights

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
    ws = weights(gf_state)

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
    using SSMProblems
    using StableRNGs
    using StatsBase: weights

    rng = StableRNG(1234)

    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, 1, 1, 1; static_arrays=true
    )
    _, _, ys = sample(rng, full_model, 4)

    resampler = GeneralisedFilters.GFTest.AlternatingResampler()
    bf = BF(10^6; resampler=resampler)
    rbbf = RBPF(bf, KalmanFilter())

    rbbf_state, llrbbf = GeneralisedFilters.filter(rng, hier_model, rbbf, ys)
    xs = getfield.(getfield.(rbbf_state.particles, :state), :x)
    zs = getfield.(getfield.(rbbf_state.particles, :state), :z)
    ws = weights(rbbf_state)

    kf_state, llkf = GeneralisedFilters.filter(rng, full_model, KF(), ys)

    @test first(kf_state.μ) ≈ sum(only.(xs) .* ws) rtol = 1e-3
    @test last(kf_state.μ) ≈ sum(only.(getfield.(zs, :μ)) .* ws) rtol = 1e-3
    @test llkf ≈ llrbbf atol = 1e-3
end

@testitem "Rao-Blackwellised GF test" begin
    using SSMProblems
    using StableRNGs
    using StatsBase: weights

    rng = StableRNG(1234)

    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, 1, 1, 1; static_arrays=true
    )
    _, _, ys = sample(rng, full_model, 4)

    prop = GeneralisedFilters.GFTest.OverdispersedProposal(dyn(hier_model).outer_dyn, 1.5)
    resampler = GeneralisedFilters.GFTest.AlternatingResampler()
    gf = ParticleFilter(10^6, prop; resampler=resampler)
    rbgf = RBPF(gf, KalmanFilter())
    rbgf_state, llrbgf = GeneralisedFilters.filter(rng, hier_model, rbgf, ys)
    xs = getfield.(getfield.(rbgf_state.particles, :state), :x)
    zs = getfield.(getfield.(rbgf_state.particles, :state), :z)
    ws = weights(rbgf_state)

    kf_state, llkf = GeneralisedFilters.filter(rng, full_model, KF(), ys)

    @test first(kf_state.μ) ≈ sum(only.(xs) .* ws) rtol = 1e-3
    @test last(kf_state.μ) ≈ sum(only.(getfield.(zs, :μ)) .* ws) rtol = 1e-3
    @test llkf ≈ llrbgf atol = 1e-3
end

@testitem "ABF test" begin
    using Distributions
    using GeneralisedFilters
    using LinearAlgebra
    using SSMProblems
    using StableRNGs
    using StatsBase: weights

    rng = StableRNG(1234)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(
        rng, 1, 1; static_arrays=true
    )
    _, _, ys = sample(rng, model, 4)

    # resampler = GeneralisedFilters.GFTest.AlternatingResampler()
    resampler = ESSResampler(0.8)
    bf = BF(10^6; resampler=resampler)
    abf = AuxiliaryParticleFilter(bf, MeanPredictive())
    abf_state, llabf = GeneralisedFilters.filter(rng, model, abf, ys)
    kf_state, llkf = GeneralisedFilters.filter(rng, model, KF(), ys)

    xs = getfield.(abf_state.particles, :state)
    ws = weights(abf_state)

    @test first(kf_state.μ) ≈ sum(first.(xs) .* ws) rtol = 1e-2
    @test llkf ≈ llabf atol = 1e-3
end

@testitem "ARBF test" begin
    using Distributions
    using GeneralisedFilters
    using LinearAlgebra
    using SSMProblems
    using StableRNGs
    using StatsBase: weights

    rng = StableRNG(1234)

    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, 1, 1, 1; static_arrays=true
    )
    _, _, ys = sample(rng, hier_model, 4)

    resampler = GeneralisedFilters.GFTest.AlternatingResampler()
    bf = BF(10^6; resampler=resampler)
    rbbf = RBPF(bf, KalmanFilter())
    arbf = AuxiliaryParticleFilter(rbbf, MeanPredictive())
    arbf_state, llarbf = GeneralisedFilters.filter(rng, hier_model, arbf, ys)
    xs = getfield.(getfield.(arbf_state.particles, :state), :x)
    zs = getfield.(getfield.(arbf_state.particles, :state), :z)
    ws = weights(arbf_state)

    kf_state, llkf = GeneralisedFilters.filter(rng, full_model, KF(), ys)

    @test first(kf_state.μ) ≈ sum(only.(xs) .* ws) rtol = 1e-2
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
    rbpf = RBPF(BF(N_particles; threshold=0.8), KalmanFilter())
    GeneralisedFilters.filter(rng, hier_model, rbpf, ys; callback=cb)

    # TODO: add proper test comparing to dense storage
    tree = cb.tree
    paths = GeneralisedFilters.get_ancestry(tree)
end

@testitem "BF on hierarchical model test" begin
    using SSMProblems
    using StableRNGs
    using StatsBase: weights

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
    ws = weights(states)

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
    using Random: randexp, AbstractRNG
    using StatsBase: sample, Weights

    struct DummyResampler <: GeneralisedFilters.AbstractResampler end

    function GeneralisedFilters.sample_ancestors(
        ::AbstractRNG, ::DummyResampler, weights::AbstractVector, n::Int64=length(weights)
    )
        return [mod1(a - 1, length(weights)) for a in 1:n]
    end

    SEED = 1234
    K = 5
    N_particles = max(10, K + 2)

    rng = StableRNG(SEED)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, 1, 1)
    _, _, ys = sample(rng, model, K)

    # Create reference trajectory
    ref_states = [rand(rng, 1) for _ in 0:K]
    ref_traj = ReferenceTrajectory(ref_states[1], ref_states[2:end])

    bf = BF(N_particles; threshold=1.0, resampler=DummyResampler())
    cb = GeneralisedFilters.DenseAncestorCallback()
    bf_state, _ = GeneralisedFilters.filter(
        rng, model, bf, ys; ref_state=ref_traj, callback=cb
    )

    traj = GeneralisedFilters.get_ancestry(cb.container, N_particles)

    # Construct expected trajectory manually
    true_x0 = cb.container.x0s[N_particles - K]
    true_xs = [cb.container.xs[t][N_particles - K + t] for t in 1:K]

    @test traj.x0 == true_x0
    @test traj.xs == true_xs

    # Test that particle 1 retrieves the reference trajectory
    ref_traj_reconstructed = GeneralisedFilters.get_ancestry(cb.container, 1)
    @test ref_traj_reconstructed.x0 == ref_traj.x0
    @test ref_traj_reconstructed.xs == ref_traj.xs
end

@testitem "CSMC test" begin
    using GeneralisedFilters
    using StableRNGs
    using PDMats
    using LinearAlgebra
    using LogExpFunctions: logsumexp
    using Random: randexp
    using StatsBase: sample, weights

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
    ref_traj = nothing
    trajectory_samples = []
    lls = []

    for i in 1:N_steps
        cb = GeneralisedFilters.DenseAncestorCallback()
        bf_state, ll = GeneralisedFilters.filter(
            rng, model, bf, ys; ref_state=ref_traj, callback=cb
        )
        ws = weights(bf_state)
        sampled_idx = sample(rng, 1:N_particles, ws)
        global ref_traj = GeneralisedFilters.get_ancestry(cb.container, sampled_idx)
        if i > N_burnin
            push!(trajectory_samples, ref_traj)
            push!(lls, ll)
        end
    end

    # The CSMC estimate of the evidence Z = p(y_{1:T}) is biased but 1 / ̂Z is actually an
    # unbiased estimate of 1 / Z. See Elements of Sequential Monte Carlo (Section 5.2)
    log_recip_likelihood_estimate = logsumexp(-lls) - log(length(lls))

    csmc_mean = sum([traj.xs[t_smooth] for traj in trajectory_samples]) / N_sample
    @test csmc_mean ≈ state.μ rtol = 1e-3
    @test log_recip_likelihood_estimate ≈ -ks_ll rtol = 1e-3
end

@testitem "RBCSMC test" begin
    using GeneralisedFilters
    using StableRNGs
    using PDMats
    using LinearAlgebra
    using Random: randexp
    using StatsBase: sample, weights
    using StaticArrays
    using Statistics

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
    ref_traj = nothing
    trajectory_samples = []

    cb = GeneralisedFilters.DenseAncestorCallback()
    for i in 1:N_steps
        bf_state, _ = GeneralisedFilters.filter(
            rng, hier_model, rbpf, ys; ref_state=ref_traj, callback=cb
        )
        ws = weights(bf_state)
        sampled_idx = sample(rng, 1:N_particles, ws)

        global ref_traj = GeneralisedFilters.get_ancestry(cb.container, sampled_idx)
        if i > N_burnin
            push!(trajectory_samples, deepcopy(ref_traj))
        end
        # Reference trajectory should only be nonlinear state for RBPF
        # Extract outer states from the hierarchical states
        ref_traj = ReferenceTrajectory(
            getproperty(ref_traj.x0, :x), getproperty.(ref_traj.xs, :x)
        )
    end

    # Extract inner and outer trajectories at time t_smooth
    x_trajectories = [getproperty(traj.xs[t_smooth], :x) for traj in trajectory_samples]

    # Manually perform smoothing until we have a cleaner interface
    A = hier_model.inner_model.dyn.A
    b = hier_model.inner_model.dyn.b
    C = hier_model.inner_model.dyn.C
    Q = hier_model.inner_model.dyn.Q
    z_smoothed_means = Vector{T}(undef, N_sample)
    for i in 1:N_sample
        μ = trajectory_samples[i].xs[K].z.μ
        Σ = trajectory_samples[i].xs[K].z.Σ

        for t in (K - 1):-1:t_smooth
            μ_filt = trajectory_samples[i].xs[t].z.μ
            Σ_filt = trajectory_samples[i].xs[t].z.Σ
            μ_pred = A * μ_filt + b + C * trajectory_samples[i].xs[t].x
            Σ_pred = X_A_Xt(Σ_filt, A) + Q
            Σ_pred = PDMat(Symmetric(Σ_pred))

            G = Σ_filt * A' / Σ_pred
            μ = μ_filt .+ G * (μ .- μ_pred)
            Σ = Σ_filt .+ G * (Σ .- Σ_pred) * G'
        end

        z_smoothed_means[i] = only(μ)
    end

    # Compare to ground truth
    @test state.μ[1] ≈ only(mean(x_trajectories)) rtol = 1e-2
    @test state.μ[2] ≈ mean(z_smoothed_means) rtol = 1e-3
end

@testitem "RBCSMC-AS test" begin
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
    ref_traj = nothing
    predictive_likelihoods = Vector{InformationLikelihood{Vector{T},Matrix{T}}}(undef, K)
    trajectory_samples = []

    for i in 1:N_steps
        global predictive_likelihoods
        cb = GeneralisedFilters.DenseAncestorCallback()

        # Manual filtering with ancestor resampling
        bf_state = initialise(rng, prior(hier_model), rbpf; ref_state=ref_traj)

        # Post-Init callback
        cb(hier_model, rbpf, bf_state, ys, PostInit)

        for t in 1:K
            bf_state = resample(rng, resampler(rbpf), bf_state; ref_state=ref_traj)

            if !isnothing(ref_traj)
                ancestor_weights = map(bf_state.particles) do particle
                    GeneralisedFilters.log_weight(particle) + ancestor_weight(
                        dyn(hier_model),
                        rbpf,
                        t,
                        particle.state,
                        RBState(ref_traj.xs[t], predictive_likelihoods[t]),
                    )
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
                bf_state.particles[end].ancestor = ancestor_idx
            end

            # Manually trigger callback
            cb(hier_model, rbpf, t, bf_state, ys[t], PostUpdate)
        end

        ws = weights(bf_state)
        sampled_idx = sample(rng, 1:N_particles, ws)

        global ref_traj = GeneralisedFilters.get_ancestry(cb.container, sampled_idx)
        if i > N_burnin
            push!(trajectory_samples, deepcopy(ref_traj))
        end
        # Reference trajectory should only be nonlinear state for RBPF
        # Extract outer states from the hierarchical states
        ref_traj_outer = ReferenceTrajectory(
            getproperty(ref_traj.x0, :x), getproperty.(ref_traj.xs, :x)
        )

        pred_lik = backward_initialise(
            rng, hier_model.inner_model.obs, BackwardInformationPredictor(), K, ys[K]
        )
        predictive_likelihoods[K] = deepcopy(pred_lik)
        for t in (K - 1):-1:1
            pred_lik = backward_predict(
                rng,
                hier_model.inner_model.dyn,
                BackwardInformationPredictor(),
                t,
                pred_lik;
                prev_outer=ref_traj_outer.xs[t],
                next_outer=ref_traj_outer.xs[t + 1],
            )
            pred_lik = backward_update(
                hier_model.inner_model.obs,
                BackwardInformationPredictor(),
                t,
                pred_lik,
                ys[t],
            )
            predictive_likelihoods[t] = deepcopy(pred_lik)
        end

        # Update ref_traj for next iteration
        global ref_traj = ref_traj_outer
    end

    # Extract inner and outer trajectories at time t_smooth
    x_trajectories = [getproperty(traj.xs[t_smooth], :x) for traj in trajectory_samples]

    # Manually perform smoothing until we have a cleaner interface
    A = hier_model.inner_model.dyn.A
    b = hier_model.inner_model.dyn.b
    C = hier_model.inner_model.dyn.C
    Q = hier_model.inner_model.dyn.Q
    z_smoothed_means = Vector{T}(undef, N_sample)
    for i in 1:N_sample
        μ = trajectory_samples[i].xs[K].z.μ
        Σ = trajectory_samples[i].xs[K].z.Σ

        for t in (K - 1):-1:t_smooth
            μ_filt = trajectory_samples[i].xs[t].z.μ
            Σ_filt = trajectory_samples[i].xs[t].z.Σ
            μ_pred = A * μ_filt + b + C * trajectory_samples[i].xs[t].x
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
