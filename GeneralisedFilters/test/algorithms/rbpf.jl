"""Unit tests for Rao-Blackwellised particle filter algorithms."""

## RBPF with Kalman Inner Filter ############################################################

@testitem "RBPF Kalman inner" begin
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

@testitem "RBPF guided Kalman inner" begin
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

@testitem "ARBF" begin
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

## RBPF with Discrete Inner Filter ##########################################################

@testitem "RBPF discrete inner" begin
    using GeneralisedFilters
    using SSMProblems
    using StableRNGs
    using StatsBase: weights

    SEED = 1234
    K_outer = 3
    K_inner = 4
    T = 10
    N_particles = 10^5

    rng = StableRNG(SEED)

    joint_model, hier_model = GeneralisedFilters.GFTest.create_dummy_discrete_model(
        rng, K_outer, K_inner; obs_separation=3.0, obs_noise=0.3
    )

    # Sample observations from the hierarchical model
    _, _, _, _, observations = sample(rng, hier_model, T)

    # Run joint forward algorithm on the product space
    joint_state, joint_ll = GeneralisedFilters.filter(rng, joint_model, DF(), observations)

    # Run RBPF with discrete inner filter
    bf = BF(N_particles)
    rbpf = RBPF(bf, DiscreteFilter())
    rbpf_state, rbpf_ll = GeneralisedFilters.filter(rng, hier_model, rbpf, observations)

    # Compare log-likelihoods
    @test joint_ll ≈ rbpf_ll atol = 0.05

    # Extract marginals from RBPF
    ws = weights(rbpf_state)
    outer_states = getfield.(getfield.(rbpf_state.particles, :state), :x)
    inner_dists = getfield.(getfield.(rbpf_state.particles, :state), :z)

    # Compute marginal outer distribution from RBPF
    rbpf_outer_marginal = zeros(K_outer)
    for (x, w) in zip(outer_states, ws)
        rbpf_outer_marginal[x] += w
    end

    # Compute marginal outer distribution from joint
    joint_outer_marginal = zeros(K_outer)
    for i in 1:K_outer
        for k in 1:K_inner
            idx = (i - 1) * K_inner + k
            joint_outer_marginal[i] += joint_state[idx]
        end
    end

    @test rbpf_outer_marginal ≈ joint_outer_marginal rtol = 0.02

    # Compute marginal inner distribution from RBPF (weighted average of inner distributions)
    rbpf_inner_marginal = zeros(K_inner)
    for (z, w) in zip(inner_dists, ws)
        rbpf_inner_marginal .+= w .* z
    end

    # Compute marginal inner distribution from joint
    joint_inner_marginal = zeros(K_inner)
    for i in 1:K_outer
        for k in 1:K_inner
            idx = (i - 1) * K_inner + k
            joint_inner_marginal[k] += joint_state[idx]
        end
    end

    @test rbpf_inner_marginal ≈ joint_inner_marginal rtol = 0.02
end

## BF on Hierarchical Models ################################################################

@testitem "BF on hierarchical model" begin
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

    # Bootstrap filter on hierarchical model (without Rao-Blackwellisation)
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

## Ancestry Tracking ########################################################################

@testitem "RBPF ancestry" begin
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

    tree = cb.tree
    paths = GeneralisedFilters.get_ancestry(tree)

    # Verify we can retrieve ancestry for all particles
    @test length(paths) == N_particles
end

@testitem "Dense ancestry" begin
    using GeneralisedFilters
    using StableRNGs
    using PDMats
    using LinearAlgebra
    using Random: randexp, AbstractRNG
    using StatsBase: sample, Weights

    using OffsetArrays

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
