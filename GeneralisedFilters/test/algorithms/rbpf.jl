"""Unit tests for Rao-Blackwellised particle filter algorithms."""

## RBPF with Kalman Inner Filter ############################################################

@testitem "RBPF Kalman inner" begin
    using GeneralisedFilters
    using Distributions
    using GeneralisedFilters
    using LinearAlgebra
    using StableRNGs
    using StatsBase: weights

    rng = StableRNG(1234)

    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, 1, 1, 1; static_arrays=true
    )
    _, _, ys = simulate(rng, full_model, 4)

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
    using GeneralisedFilters
    using StableRNGs
    using StatsBase: weights

    rng = StableRNG(1234)

    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, 1, 1, 1; static_arrays=true
    )
    _, _, ys = simulate(rng, full_model, 4)

    prop = GeneralisedFilters.GFTest.OverdispersedProposal(hier_model.dyn.outer, 1.5)
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
    using GeneralisedFilters
    using Distributions
    using GeneralisedFilters
    using LinearAlgebra
    using StableRNGs
    using StatsBase: weights

    rng = StableRNG(1234)

    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, 1, 1, 1; static_arrays=true
    )
    _, _, ys = simulate(rng, hier_model, 4)

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
    using GeneralisedFilters
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
    _, _, observations = simulate(rng, hier_model, T)

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
    using GeneralisedFilters
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
    _, _, ys = simulate(rng, full_model, T)

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
    using GeneralisedFilters
    using StableRNGs

    SEED = 1234
    T = 5
    N_particles = 100

    rng = StableRNG(SEED)
    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, 1, 1, 1
    )
    _, _, ys = simulate(rng, full_model, T)

    rbpf = RBPF(BF(N_particles; threshold=0.8), KalmanFilter())
    initial = GeneralisedFilters.initialise(rng, hier_model.prior, rbpf)
    state, _ = GeneralisedFilters.step(rng, hier_model, rbpf, 1, initial, ys[1])
    tree = GeneralisedFilters._init_tree(initial, state)
    for t in 2:T
        global state, _ = GeneralisedFilters.step(rng, hier_model, rbpf, t, state, ys[t])
        GeneralisedFilters._update_tree!(tree, state)
    end
    paths = GeneralisedFilters.get_ancestry(tree)

    # Verify we can retrieve ancestry for all particles
    @test length(paths) == N_particles
end

@testitem "Dense ancestry" begin
    using GeneralisedFilters
    using GeneralisedFilters
    using StableRNGs
    using PDMats
    using LinearAlgebra
    using Random: randexp, AbstractRNG
    using StatsBase: sample, Weights

    using GeneralisedFilters: ReferenceTrajectory

    struct DummyResampler <: GeneralisedFilters.AbstractResampler end

    function GeneralisedFilters.sample_ancestors(
        ::AbstractRNG, ::DummyResampler, weights::AbstractVector, n::Int64=length(weights)
    )
        return [mod1(a - 1, length(weights)) for a in 1:n]
    end

    GeneralisedFilters.supports_conditional(::DummyResampler) = true

    function GeneralisedFilters.conditional_sample_ancestors(
        ::AbstractRNG, ::DummyResampler, weights::AbstractVector, ref_idx::Integer
    )
        n = length(weights)
        return [a == 1 ? ref_idx : mod1(a - 1, n) for a in 1:n]
    end

    SEED = 1234
    K = 5
    N_particles = max(10, K + 2)

    rng = StableRNG(SEED)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, 1, 1)
    _, _, ys = simulate(rng, model, K)

    ref_traj = ReferenceTrajectory(rand(rng, 1), [rand(rng, 1) for _ in 1:K])

    bf = BF(N_particles; threshold=1.0, resampler=DummyResampler())
    initial = GeneralisedFilters.initialise(rng, model.prior, bf; ref_state=ref_traj)
    state, _ = GeneralisedFilters.step(
        rng, model, bf, 1, initial, ys[1]; ref_state=ref_traj
    )
    container = GeneralisedFilters._init_container(initial, state)
    for t in 2:K
        global state, _ = GeneralisedFilters.step(
            rng, model, bf, t, state, ys[t]; ref_state=ref_traj
        )
        GeneralisedFilters._update_container!(container, state)
    end

    traj = GeneralisedFilters.get_ancestry(container, N_particles)
    true_x0 = container.initial_states[N_particles - K]
    true_xs = [container.states[t][N_particles - K + t] for t in 1:K]

    @test traj.x0 == true_x0
    @test traj.xs == true_xs
    @test GeneralisedFilters.get_ancestry(container, 1) == ref_traj
end

@testitem "RB proposal sees filtering belief and returns outer state" begin
    using GeneralisedFilters
    using StableRNGs
    using StaticArrays
    using Distributions
    const GF = GeneralisedFilters

    struct BeliefProposal <: AbstractProposal end
    GF.distribution(::BeliefProposal, t::Integer, s::RBState, y) =
        GaussianState(s.x + s.z.μ, s.z.Σ)

    rng = StableRNG(97)
    outer = LinearGaussianDynamics(@SMatrix([0.8;;]), @SVector([0.1]), @SMatrix([0.4;;]))
    inner = ctx -> LinearGaussianDynamics(@SMatrix([0.7;;]), ctx.x_new, @SMatrix([0.2;;]))
    dynamics = HierarchicalDynamics(outer, inner)
    state = RBState(@SVector([0.3]), GaussianState(@SVector([0.5]), @SMatrix([0.6;;])))
    particle = GF.Particle(state, -0.2, 3)
    algo = RBPF(ParticleFilter(4, BeliefProposal()), KF())
    xnew = @SVector([1.1])
    y = @SVector([0.4])
    result = GF.predict_particle(rng, dynamics, algo, 1, particle, y, xnew)
    expected_weight =
        -0.2 + logdensity(outer, 1, state.x, xnew) -
        logpdf(GF.distribution(BeliefProposal(), 1, state, y), xnew)
    @test GF.log_weight(result) ≈ expected_weight
    @test result.state.x == xnew
    @test result.state.z.μ ≈ 0.7 * state.z.μ + xnew
    @test result.ancestor == 3

    # Reinitialisation must use the new conditional prior, not any stored inner belief.
    prior = HierarchicalPrior(
        DistributionPrior(Normal()),
        ctx -> GaussianPrior(@SVector([ctx.x0]), @SMatrix([0.5;;])),
    )
    initial = GF.initialise_particle(rng, prior, algo, 2.0)
    @test initial.state.x == 2.0
    @test initial.state.z.μ == @SVector([2.0])
end
