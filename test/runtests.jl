using TestItems
using TestItemRunner

@run_package_tests

include("batch_kalman_test.jl")
include("resamplers.jl")

@testitem "Kalman filter test" setup = [TestModels] begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs

    SEED = 1234
    Dx = 3
    Dys = [2, 3, 4]

    for Dy in Dys
        rng = StableRNG(1234)
        model = TestModels.create_linear_gaussian_model(rng, Dx, Dy)
        _, _, ys = sample(rng, model, 1)

        states, ll = GeneralisedFilters.filter(rng, model, KalmanFilter(), ys)

        # Let Z = [X0, X1, Y1] be the joint state vector
        μ_Z, Σ_Z = TestModels._compute_joint(model, 1)

        # Condition on observations using formula for MVN conditional distribution. See: 
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        y = only(ys)
        I_x = (Dx + 1):(2Dx)
        I_y = (2Dx + 1):(2Dx + Dy)
        μ_X1 = μ_Z[I_x] + Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ (y - μ_Z[I_y]))
        Σ_X1 = Σ_Z[I_x, I_x] - Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ Σ_Z[I_y, I_x])

        @test states.filtered.μ ≈ μ_X1
        @test states.filtered.Σ ≈ Σ_X1

        # Exact marginal distribution to test log-likelihood
        μ_Y1 = μ_Z[I_y]
        Σ_Y1 = Σ_Z[I_y, I_y]
        LinearAlgebra.hermitianpart!(Σ_Y1)
        true_ll = logpdf(MvNormal(μ_Y1, Σ_Y1), y)
        @test ll ≈ true_ll
    end
end

@testitem "Kalman smoother test" setup = [TestModels] begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs

    SEED = 1234
    Dx = 3
    Dys = [2, 3, 4]

    for Dy in Dys
        rng = StableRNG(1234)
        model = TestModels.create_linear_gaussian_model(rng, Dx, Dy)
        _, _, ys = sample(rng, model, 2)

        states, ll = GeneralisedFilters.smooth(rng, model, KalmanSmoother(), ys)

        # Let Z = [X0, X1, X2, Y1, Y2] be the joint state vector
        μ_Z, Σ_Z = TestModels._compute_joint(model, 2)

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
    using GeneralisedFilters
    using SSMProblems
    using StableRNGs
    using PDMats
    using LinearAlgebra
    using LogExpFunctions: softmax
    using Random: randexp

    T = Float32
    rng = StableRNG(1234)
    σx², σy² = randexp(rng, T, 2)

    # initial state distribution
    μ0 = zeros(T, 2)
    Σ0 = PDMat(T[1 0; 0 1])

    # state transition equation
    A = T[1 1; 0 1]
    b = T[0; 0]
    Q = PDiagMat([σx²; 0])

    # observation equation
    H = T[1 0]
    c = T[0;]
    R = [σy²;;]

    # when working with PDMats, the Kalman filter doesn't play nicely without this
    function Base.convert(::Type{PDMat{T,MT}}, mat::MT) where {MT<:AbstractMatrix,T<:Real}
        return PDMat(Symmetric(mat))
    end

    model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
    _, _, data = sample(rng, model, 20)

    bf = BF(2^12; threshold=0.8)
    bf_state, llbf = GeneralisedFilters.filter(rng, model, bf, data)
    kf_state, llkf = GeneralisedFilters.filter(rng, model, KF(), data)

    xs = bf_state.filtered.particles
    ws = softmax(bf_state.filtered.log_weights)

    # Compare filtered states
    @test first(kf_state.filtered.μ) ≈ sum(first.(xs) .* ws) rtol = 1e-2

    # since this is log valued, we can up the tolerance
    @test llkf ≈ llbf atol = 0.1
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

    struct MixtureModelObservation{T} <: SSMProblems.ObservationProcess{T}
        μs::Vector{T}
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

    dyn = HomogeneousDiscreteLatentDynamics{Int,Float64}(α0, P)
    obs = MixtureModelObservation(μs)
    model = StateSpaceModel(dyn, obs)

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

@testitem "Kalman-RBPF test" setup = [TestModels] begin
    using LogExpFunctions: softmax
    using SSMProblems
    using StableRNGs

    SEED = 1234
    D_outer = 2
    D_inner = 3
    D_obs = 2
    T = 5
    N_particles = 10^4

    rng = StableRNG(SEED)

    full_model, hier_model = TestModels.create_dummy_linear_gaussian_model(
        rng, D_outer, D_inner, D_obs
    )
    _, _, ys = sample(rng, full_model, T)

    # Ground truth Kalman filtering
    kf_states, kf_ll = GeneralisedFilters.filter(rng, full_model, KalmanFilter(), ys)

    # Rao-Blackwellised particle filtering
    rbpf = BatchRBPF(BatchKalmanFilter(N_particles), N_particles)
    states, ll = GeneralisedFilters.filter(rng, hier_model, rbpf, ys)

    # Extract final filtered states
    xs = map(p -> getproperty(p, :x), states.filtered.particles)
    zs = map(p -> getproperty(p, :z), states.filtered.particles)
    log_ws = states.filtered.log_weights

    @test kf_ll ≈ ll rtol = 1e-2

    weights = Weights(softmax(log_ws))

    # Higher tolerance for outer state since variance is higher
    @test first(kf_states.μ) ≈ sum(first.(xs) .* weights) rtol = 1e-1
    @test last(kf_states.μ) ≈ sum(last.(getproperty.(zs, :μ)) .* weights) rtol = 1e-2
end

@testitem "GPU Kalman-RBPF test" setup = [TestModels] begin
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

    rng = StableRNG(SEED)

    full_model, hier_model = TestModels.create_dummy_linear_gaussian_model(
        rng, D_outer, D_inner, D_obs
    )
    _, _, ys = sample(rng, full_model, T)

    # Ground truth Kalman filtering
    kf_state, kf_ll = GeneralisedFilters.filter(full_model, KalmanFilter(), ys)

    # Rao-Blackwellised particle filtering
    rbpf = BatchRBPF(BatchKalmanFilter(N_particles), N_particles)
    states, ll = GeneralisedFilters.filter(hier_model, rbpf, ys)

    # Extract final filtered states
    xs = states.filtered.particles.x_particles
    zs = states.filtered.particles.z_particles
    log_ws = states.filtered.log_weights

    weights = softmax(log_ws)
    reshaped_weights = reshape(weights, (1, length(weights)))

    @test kf_ll ≈ ll rtol = 1e-2
    @test first(kf_state.filtered.μ) ≈ sum(xs[1, :] .* weights) rtol = 1e-1
    @test last(kf_state.filtered.μ) ≈ sum(zs.μs[end, :] .* weights) rtol = 1e-2
end

@testitem "RBPF ancestory test" setup = [TestModels] begin
    using SSMProblems
    using StableRNGs

    SEED = 1234
    T = 5
    N_particles = 100

    rng = StableRNG(SEED)
    full_model, hier_model = TestModels.create_dummy_linear_gaussian_model(rng, 1, 1, 1)
    _, _, ys = sample(rng, full_model, T)

    # Manually create tree to force expansion on second step
    particle_type = GeneralisedFilters.RaoBlackwellisedContainer{
        eltype(hier_model.outer_dyn),GeneralisedFilters.rb_eltype(hier_model.inner_model)
    }
    nodes = Vector{particle_type}(undef, N_particles)
    tree = GeneralisedFilters.ParticleTree(nodes, N_particles + 1)

    rbpf = RBPF(KalmanFilter(), N_particles)
    cb = GeneralisedFilters.AncestorCallback(tree)
    GeneralisedFilters.filter(rng, hier_model, rbpf, ys; callback=cb)

    # TODO: add proper test comparing to dense storage
    tree = cb.tree
    paths = GeneralisedFilters.get_ancestry(tree)
end

@testitem "CSMC test" setup = [TestModels] begin
    using GeneralisedFilters
    using SSMProblems
    using StableRNGs
    using PDMats
    using LinearAlgebra
    using LogExpFunctions: softmax
    using Random: randexp
    using StatsBase: sample, Weights

    using OffsetArrays

    SEED = 1234
    Dx = 1
    Dy = 1
    K = 5
    t_smooth = 2
    T = Float64
    N_particles = 1000
    N_burnin = 100
    N_sample = 500

    rng = StableRNG(SEED)
    model = TestModels.create_linear_gaussian_model(rng, Dx, Dy)
    _, _, ys = sample(rng, model, K)

    # Kalman smoother
    state, _ = GeneralisedFilters.smooth(
        rng, model, KalmanSmoother(), ys; t_smooth=t_smooth
    )

    N_steps = N_burnin + N_sample
    bf = BF(N_particles)
    ref_traj = nothing
    trajectory_samples = Vector{OffsetVector{Vector{T},Vector{Vector{T}}}}(undef, N_sample)

    for i in 1:N_steps
        cb = GeneralisedFilters.DenseAncestorCallback(Vector{T})
        bf_state, _ = GeneralisedFilters.filter(
            rng, model, bf, ys; ref_state=ref_traj, callback=cb
        )
        weights = softmax(bf_state.filtered.log_weights)
        sampled_idx = sample(rng, 1:length(weights), Weights(weights))
        global ref_traj = GeneralisedFilters.get_ancestry(cb.container, sampled_idx)
        if i > N_burnin
            trajectory_samples[i - N_burnin] = ref_traj
        end
    end

    csmc_mean = sum(getindex.(trajectory_samples, t_smooth)) / N_sample
    @test csmc_mean ≈ state.μ rtol = 1e-1
end

@testitem "RBCSMC test" begin
    using GeneralisedFilters
    using SSMProblems
    using GaussianDistributions
    using StableRNGs
    using PDMats
    using LinearAlgebra
    using LogExpFunctions: softmax
    using Random: randexp
    using StatsBase: sample, Weights

    D_outer = 1
    D_inner = 1
    D_obs = 1

    # Define inner dynamics
    struct InnerDynamics{T} <: LinearGaussianLatentDynamics{T}
        μ0::Vector{T}
        Σ0::Matrix{T}
        A::Matrix{T}
        b::Vector{T}
        C::Matrix{T}
        Q::Matrix{T}
    end
    GeneralisedFilters.calc_μ0(dyn::InnerDynamics; kwargs...) = dyn.μ0
    GeneralisedFilters.calc_Σ0(dyn::InnerDynamics; kwargs...) = dyn.Σ0
    GeneralisedFilters.calc_A(dyn::InnerDynamics, ::Integer; kwargs...) = dyn.A
    function GeneralisedFilters.calc_b(dyn::InnerDynamics, ::Integer; prev_outer, kwargs...)
        return dyn.b + dyn.C * prev_outer
    end
    GeneralisedFilters.calc_Q(dyn::InnerDynamics, ::Integer; kwargs...) = dyn.Q

    rng = StableRNG(1234)
    μ0 = rand(rng, D_outer + D_inner)
    Σ0s = [rand(rng, D_outer, D_outer), rand(rng, D_inner, D_inner)]
    Σ0s = [Σ * Σ' for Σ in Σ0s]  # make Σ0 positive definite
    Σ0 = [
        Σ0s[1] zeros(D_outer, D_inner)
        zeros(D_inner, D_outer) Σ0s[2]
    ]
    A = [
        rand(rng, D_outer, D_outer) zeros(D_outer, D_inner)
        rand(rng, D_inner, D_outer + D_inner)
    ]
    # Make mean-reverting
    A /= 3.0
    A[diagind(A)] .= -0.5
    b = rand(rng, D_outer + D_inner)
    Qs = [rand(rng, D_outer, D_outer), rand(rng, D_inner, D_inner)] ./ 10.0
    Qs = [Q * Q' for Q in Qs]  # make Q positive definite
    Q = [
        Qs[1] zeros(D_outer, D_inner)
        zeros(D_inner, D_outer) Qs[2]
    ]
    H = [zeros(D_obs, D_outer) rand(rng, D_obs, D_inner)]
    c = rand(rng, D_obs)
    R = rand(rng, D_obs, D_obs)
    R = R * R' / 3.0  # make R positive definite

    outer_dyn = GeneralisedFilters.HomogeneousLinearGaussianLatentDynamics(
        μ0[1:D_outer],
        Σ0[1:D_outer, 1:D_outer],
        A[1:D_outer, 1:D_outer],
        b[1:D_outer],
        Qs[1],
    )
    inner_dyn = InnerDynamics(
        μ0[(D_outer + 1):end],
        Σ0[(D_outer + 1):end, (D_outer + 1):end],
        A[(D_outer + 1):end, (D_outer + 1):end],
        b[(D_outer + 1):end],
        A[(D_outer + 1):end, 1:D_outer],
        Qs[2],
    )
    obs = GeneralisedFilters.HomogeneousLinearGaussianObservationProcess(
        H[:, (D_outer + 1):end], c, R
    )
    model = HierarchicalSSM(outer_dyn, inner_dyn, obs)

    T = GeneralisedFilters.RaoBlackwellisedContainer{
        Vector{Float64},Gaussian{Vector{Float64},Matrix{Float64}}
    }
    data = [rand(rng, D_obs) for _ in 1:5]

    # Naive smoother
    N_particles_1 = 100000
    cb = GeneralisedFilters.DenseAncestorCallback(T)
    rbpf = RBPF(KalmanFilter(), N_particles_1; threshold=0.8)
    bf_state, llbf = GeneralisedFilters.filter(rng, model, rbpf, data; callback=cb)
    weights = softmax(bf_state.filtered.log_weights)
    sampled_indices = sample(rng, 1:length(weights), Weights(weights), N_particles_1)
    μs = Vector{T}(undef, N_particles_1)
    for i in 1:N_particles_1
        μs[i] = GeneralisedFilters.get_ancestry(cb.container, sampled_indices[i])[3]
    end

    N_particles = 1000
    cb = GeneralisedFilters.DenseAncestorCallback(T)

    # TODO: re-introduce resampling and trace ancestory
    let
        rbpf = RBPF(KalmanFilter(), N_particles; threshold=0.8, resampler=Multinomial())
        bf_state, llbf = GeneralisedFilters.filter(rng, model, rbpf, data; callback=cb)
        weights = softmax(bf_state.filtered.log_weights)
        sampled_idx = sample(rng, 1:length(weights), Weights(weights))
        ref_traj = GeneralisedFilters.get_ancestry(cb.container, sampled_idx)

        N_burnin = 100
        N_sample = 2000
        N_steps = N_burnin + N_sample
        trajectory_samples = Vector{Vector{T}}(undef, N_sample)
        for i in 1:N_steps
            bf_state, _ = GeneralisedFilters.filter(
                rng, model, rbpf, data; ref_state=ref_traj
            )
            weights = softmax(bf_state.filtered.log_weights)
            sampled_idx = sample(rng, 1:length(weights), Weights(weights))
            ref_traj = GeneralisedFilters.get_ancestry(cb.container, sampled_idx)
            if i > N_burnin
                trajectory_samples[i - N_burnin] = ref_traj
            end
        end

        # Compare smoothed states
        naive_mean = first.(sum(getproperty.(μs, :x)) / N_particles_1)
        csmc_mean =
            first.(sum(getproperty.(getindex.(trajectory_samples, 3), :x)) / N_sample)
        @test csmc_mean[1] ≈ naive_mean[1] rtol = 1e-1
    end
end

@testitem "GPU Conditional Kalman-RBPF execution test" setup = [TestModels] begin
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

    full_model, hier_model = TestModels.create_dummy_linear_gaussian_model(
        rng, D_outer, D_inner, D_obs, T
    )
    _, _, ys = sample(rng, full_model, K)

    # Generate random reference trajectory
    ref_trajectory = [
        GeneralisedFilters.RaoBlackwellisedParticleState(
            GeneralisedFilters.RaoBlackwellisedParticle(
                CuArray(rand(rng, T, D_outer, 1)),
                GeneralisedFilters.BatchGaussianDistribution(
                    CuArray(rand(rng, T, D_inner, 1)),
                    CuArray(reshape(TestModels.rand_cov(rng, T, D_inner), Val(3))),
                ),
            ),
            CUDA.zeros(T, 1),  # arbitrary log weight
        ) for _ in 0:K
    ]
    ref_trajectory = OffsetVector(ref_trajectory, -1)

    rbpf = BatchRBPF(BatchKalmanFilter(N_particles), N_particles)
    states, ll = GeneralisedFilters.filter(hier_model, rbpf, ys; ref_state=ref_trajectory)

    # Check returned type
    @test typeof(ll) == T
end

@testitem "GPU-RBPF ancestory test" setup = [TestModels] begin
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
    N_particles = 1000

    rng = StableRNG(1234)

    full_model, hier_model = TestModels.create_dummy_linear_gaussian_model(
        rng, D_outer, D_inner, D_obs, T
    )
    _, _, ys = sample(rng, full_model, K)

    # Manually create tree to force expansion on second step
    M = N_particles * 2 - 1
    tree = GeneralisedFilters.ParallelParticleTree(
        GeneralisedFilters.RaoBlackwellisedParticle(
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

@testitem "GPU Conditional Kalman-RBPF validity test" setup = [TestModels] begin
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
    K = 5
    t_smooth = 2
    T = Float32
    N_particles = 1000
    N_burnin = 100
    N_sample = 500

    rng = StableRNG(1234)

    full_model, hier_model = TestModels.create_dummy_linear_gaussian_model(
        rng, D_outer, D_inner, D_obs, T
    )
    _, _, ys = sample(rng, full_model, K)

    # Kalman smoother
    state, _ = GeneralisedFilters.smooth(
        rng, full_model, KalmanSmoother(), ys; t_smooth=t_smooth
    )

    particle_template = GeneralisedFilters.RaoBlackwellisedParticle(
        CuArray{Float32}(undef, D_outer, N_particles),
        GeneralisedFilters.BatchGaussianDistribution(
            CuArray{Float32}(undef, D_inner, N_particles),
            CuArray{Float32}(undef, D_inner, D_inner, N_particles),
        ),
    )
    particle_type = typeof(particle_template)

    N_steps = N_burnin + N_sample
    M = floor(Int64, N_particles * log(N_particles))
    rbpf = BatchRBPF(BatchKalmanFilter(N_particles), N_particles)
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
        weights = softmax(rbpf_state.filtered.log_weights)
        sampled_idx = CUDA.@allowscalar sample(1:length(weights), Weights(weights))
        global ref_traj = GeneralisedFilters.get_ancestry(tree, sampled_idx, K)
        if i > N_burnin
            trajectory_samples[i - N_burnin] = getproperty.(ref_traj, :particles)
        end
    end

    # Extract inner and outer trajectories
    x_trajectories = getproperty.(getindex.(trajectory_samples, t_smooth), :x_particles)
    z_trajectories = getproperty.(getindex.(trajectory_samples, t_smooth), :z_particles)

    # Compare to ground truth
    CUDA.@allowscalar begin
        @test state.μ[1] ≈ only(mean(x_trajectories)) rtol = 1e-1
        @test state.μ[2] ≈ only(mean(getproperty.(z_trajectories, :μs))) rtol = 1e-1
    end
end
