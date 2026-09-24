"""Tests for the Turing @model integration with ParticleGibbs."""

## NUTS: smoke test ############################################################################

@testitem "ParticleGibbs Turing NUTS: smoke test" begin
    using GeneralisedFilters
    using AbstractMCMC: AbstractMCMC
    using AdvancedHMC: NUTS
    using MCMCChains: MCMCChains
    using Turing
    using StableRNGs
    using Distributions
    using PDMats
    using LinearAlgebra
    using ForwardDiff

    rng = StableRNG(1234)

    a = 0.8
    q² = 0.1
    r² = 0.5
    σ₀² = 1.0
    T_len = 5
    N_particles = 20
    N_iter = 10

    function build_ssm_smoke(drift)
        return create_homogeneous_linear_gaussian_model(
            [0.0],
            PDMat([σ₀²;;]),
            [a;;],
            [drift[1]],
            PDMat([q²;;]),
            [1.0;;],
            [0.0],
            PDMat([r²;;]),
        )
    end

    true_ssm = build_ssm_smoke([1.5])
    _, _, ys = simulate(rng, true_ssm, T_len)

    @model function drift_model_smoke(ys)
        b ~ MvNormal([0.0], 4.0 * I)
        ssm = build_ssm_smoke(b)
        x ~ SSMTrajectory(ssm, ys)
        return nothing
    end

    m = drift_model_smoke(ys)
    pg = ParticleGibbs(
        ConditionalSMC(BF(N_particles; resampler=GeneralisedFilters.Multinomial())),
        NUTS(0.8),
    )

    chain = AbstractMCMC.sample(
        rng, m, pg, N_iter; n_adapts=5, progress=false, chain_type=MCMCChains.Chains
    )

    @test chain isa MCMCChains.Chains
    @test size(chain, 1) == N_iter
end

## MH: smoke test #############################################################################

@testitem "ParticleGibbs Turing MH: smoke test" begin
    using GeneralisedFilters
    using AbstractMCMC: AbstractMCMC
    using AdvancedMH: RWMH
    using MCMCChains: MCMCChains
    using Turing
    using StableRNGs
    using Distributions
    using PDMats
    using LinearAlgebra

    rng = StableRNG(1234)

    a = 0.8
    q² = 0.1
    r² = 0.5
    σ₀² = 1.0
    T_len = 5
    N_particles = 10
    N_iter = 10

    function build_ssm_mh(drift)
        return create_homogeneous_linear_gaussian_model(
            [0.0],
            PDMat([σ₀²;;]),
            [a;;],
            [drift[1]],
            PDMat([q²;;]),
            [1.0;;],
            [0.0],
            PDMat([r²;;]),
        )
    end

    true_ssm = build_ssm_mh([1.5])
    _, _, ys = simulate(rng, true_ssm, T_len)

    @model function drift_model_mh(ys)
        b ~ MvNormal([0.0], 4.0 * I)
        ssm = build_ssm_mh(b)
        x ~ SSMTrajectory(ssm, ys)
        return nothing
    end

    m = drift_model_mh(ys)
    pg = ParticleGibbs(
        ConditionalSMC(BF(N_particles; resampler=GeneralisedFilters.Multinomial())),
        RWMH(MvNormal(zeros(1), 0.5 * I)),
    )

    chain = AbstractMCMC.sample(
        rng, m, pg, N_iter; progress=false, chain_type=MCMCChains.Chains
    )

    @test chain isa MCMCChains.Chains
    @test size(chain, 1) == N_iter
    @test :accepted in names(chain, :internals)
end

## NUTS: regular SSM against augmented KF ######################################################

@testitem "ParticleGibbs Turing NUTS: regular SSM" begin
    using GeneralisedFilters
    using AbstractMCMC: AbstractMCMC
    using AdvancedHMC: NUTS
    using MCMCChains: MCMCChains
    using Turing
    using StableRNGs
    using Distributions
    using PDMats
    using LinearAlgebra
    using Statistics
    using ForwardDiff

    rng = StableRNG(42)

    a = 0.8
    q² = 0.1
    r² = 0.5
    σ₀² = 1.0
    σ_b² = 4.0
    T_len = 10
    N_particles = 50
    N_iter = 5000
    N_adapts = 500

    function build_ssm_reg(drift)
        return create_homogeneous_linear_gaussian_model(
            [0.0],
            PDMat([σ₀²;;]),
            [a;;],
            [drift[1]],
            PDMat([q²;;]),
            [1.0;;],
            [0.0],
            PDMat([r²;;]),
        )
    end

    true_b = 1.5
    true_ssm = build_ssm_reg([true_b])
    _, _, ys = simulate(rng, true_ssm, T_len)

    # Augmented KF ground truth
    ref_model = build_ssm_reg([0.0])
    kf_post = GeneralisedFilters.GFTest.augmented_kf_drift_posterior(
        ref_model, ys, 1; σ²_b=σ_b², ε=1e-12
    )
    kf_mean = kf_post.mean[1]
    kf_std = kf_post.std[1]

    @model function drift_model_reg(ys)
        b ~ MvNormal([0.0], σ_b² * I)
        ssm = build_ssm_reg(b)
        x ~ SSMTrajectory(ssm, ys)
        return nothing
    end

    m = drift_model_reg(ys)
    pg = ParticleGibbs(
        ConditionalSMC(BF(N_particles; resampler=GeneralisedFilters.Multinomial())),
        NUTS(0.8),
    )

    chain = AbstractMCMC.sample(
        rng, m, pg, N_iter; n_adapts=N_adapts, progress=false, chain_type=MCMCChains.Chains
    )

    # Find the parameter column (not trajectory)
    post_samples = Array(chain[Symbol("b[1]")])[(N_adapts + 1):end]

    @test mean(post_samples) ≈ kf_mean rtol = 0.1
    @test std(post_samples) ≈ kf_std rtol = 0.2
end

## NUTS: BF on HierarchicalSSM against augmented KF ###########################################

@testitem "ParticleGibbs Turing NUTS: BF on HierarchicalSSM" begin
    using GeneralisedFilters
    using AbstractMCMC: AbstractMCMC
    using AdvancedHMC: NUTS
    using MCMCChains: MCMCChains
    using Turing
    using StableRNGs
    using Distributions
    using PDMats
    using LinearAlgebra
    using Statistics
    using ForwardDiff

    rng = StableRNG(42)

    Dx, Dz, Dy = 1, 1, 1
    T_len = 4
    N_particles = 500
    # Small process noise strongly couples drift and sampled inner states. The original
    # 5,000-sweep chain had only about 40 effective drift draws; retain this difficult
    # full-state fixture and use enough sweeps for the unchanged posterior tolerances.
    N_iter = 50000
    N_adapts = 500
    σ²_b = 4.0

    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, Dx, Dz, Dy; static_arrays=true
    )
    x0, xs, ys = simulate(rng, hier_model, T_len)

    # The full Gaussian reference and hierarchical sampled-state target must agree
    # for the same complete trajectory at several parameter values.
    states = [x0; xs]
    full_states = [vcat(s.x, s.z) for s in states]
    flat = reduce(vcat, full_states)
    for b in (-0.6, 0.0, 1.2)
        dy = full_model.dyn
        full_b = typeof(dy.b)([dy.b[1], b])
        full = StateSpaceModel(
            full_model.prior, LinearGaussianDynamics(dy.A, full_b, dy.Q), full_model.obs
        )
        hier = GeneralisedFilters.GFTest.with_inner_drift(hier_model, [b])
        expected = trajectory_logdensity(full, full_states, ys)
        @test trajectory_logdensity(hier, states, ys) ≈ expected
        @test logpdf(SSMTrajectory(hier, ys), flat) ≈ expected
    end

    fixed = hier_model

    drift_indices = (Dx + 1):(Dx + Dz)
    kf_post = GeneralisedFilters.GFTest.augmented_kf_drift_posterior(
        full_model, ys, drift_indices; σ²_b=σ²_b, ε=1e-12
    )
    kf_mean = kf_post.mean
    kf_std = kf_post.std

    @model function drift_model_bf_hier(ys)
        b ~ MvNormal(zeros(Dz), σ²_b * I)
        ssm = GeneralisedFilters.GFTest.with_inner_drift(fixed, b)
        # No inner filter: BF samples the full HierarchicalState (outer + inner)
        x ~ SSMTrajectory(ssm, ys)
        return nothing
    end

    m = drift_model_bf_hier(ys)
    pg = ParticleGibbs(
        ConditionalSMC(
            BF(N_particles; resampler=GeneralisedFilters.Multinomial()), AncestorSampling()
        ),
        NUTS(0.8),
    )

    chain = AbstractMCMC.sample(
        rng, m, pg, N_iter; n_adapts=N_adapts, progress=false, chain_type=MCMCChains.Chains
    )

    post_samples = Array(chain[Symbol("b[1]")])[(N_adapts + 1):end]

    @test mean(post_samples) ≈ kf_mean[1] rtol = 1e-1
    @test std(post_samples) ≈ kf_std[1] rtol = 1e-1
end

## NUTS: HierarchicalSSM against augmented KF ##################################################

@testitem "ParticleGibbs Turing NUTS: HierarchicalSSM" begin
    using GeneralisedFilters
    using AbstractMCMC: AbstractMCMC
    using AdvancedHMC: NUTS
    using MCMCChains: MCMCChains
    using ADTypes: ADTypes
    using Turing
    using StableRNGs
    using Distributions
    using PDMats
    using LinearAlgebra
    using Statistics
    using Mooncake, ForwardDiff

    rng = StableRNG(42)

    Dx, Dz, Dy = 1, 1, 1
    T_len = 10
    N_particles = 50
    N_iter = 5000
    N_adapts = 500
    σ²_b = 4.0

    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, Dx, Dz, Dy; static_arrays=true
    )
    _, _, ys = simulate(rng, hier_model, T_len)

    fixed = hier_model

    # Augmented KF ground truth
    drift_indices = (Dx + 1):(Dx + Dz)
    kf_post = GeneralisedFilters.GFTest.augmented_kf_drift_posterior(
        full_model, ys, drift_indices; σ²_b=σ²_b, ε=1e-12
    )
    kf_mean = kf_post.mean
    kf_std = kf_post.std

    @model function drift_model_hier(ys)
        b ~ MvNormal(zeros(Dz), σ²_b * I)
        ssm = GeneralisedFilters.GFTest.with_inner_drift(fixed, b)
        x ~ SSMTrajectory(ssm, KF(), ys)
        return nothing
    end

    m = drift_model_hier(ys)
    for backend in (ADTypes.AutoForwardDiff(), ADTypes.AutoMooncake(; config=nothing))
        pg = ParticleGibbs(
            ConditionalSMC(
                RBPF(BF(N_particles; resampler=GeneralisedFilters.Multinomial()), KF())
            ),
            NUTS(0.8);
            adtype=backend,
        )

        chain = AbstractMCMC.sample(
            rng,
            m,
            pg,
            N_iter;
            n_adapts=N_adapts,
            progress=false,
            chain_type=MCMCChains.Chains,
        )

        post_samples = Array(chain[Symbol("b[1]")])[(N_adapts + 1):end]

        @test mean(post_samples) ≈ kf_mean[1] rtol = 1e-1
        @test std(post_samples) ≈ kf_std[1] rtol = 1e-1
    end
end

## Joint NUTS: regular SSM with Mooncake #######################################################
# Runs NUTS directly on (b, x₀:T) — no ParticleGibbs alternation. Mooncake handles the
# trajectory log-density including Gaussian factorisations.

@testitem "Joint NUTS: regular SSM with Mooncake" tags = [:mooncake] begin
    using GeneralisedFilters
    using ADTypes: AutoMooncake
    using MCMCChains: MCMCChains
    using Turing
    using StableRNGs
    using Distributions
    using PDMats
    using LinearAlgebra
    using Statistics
    using Mooncake

    rng = StableRNG(42)

    a = 0.8
    q² = 0.1
    r² = 0.5
    σ₀² = 1.0
    σ_b² = 4.0
    T_len = 10
    N_iter = 2000
    N_adapts = 500

    function build_ssm_joint_reg_mooncake(drift)
        return create_homogeneous_linear_gaussian_model(
            [0.0],
            PDMat([σ₀²;;]),
            [a;;],
            [drift[1]],
            PDMat([q²;;]),
            [1.0;;],
            [0.0],
            PDMat([r²;;]),
        )
    end

    true_ssm = build_ssm_joint_reg_mooncake([1.5])
    _, _, ys = simulate(rng, true_ssm, T_len)

    ref_model = build_ssm_joint_reg_mooncake([0.0])
    kf_post = GeneralisedFilters.GFTest.augmented_kf_drift_posterior(
        ref_model, ys, 1; σ²_b=σ_b², ε=1e-12
    )

    @model function joint_drift_model_reg_mooncake(ys)
        b ~ MvNormal([0.0], σ_b² * I)
        ssm = build_ssm_joint_reg_mooncake(b)
        x ~ SSMTrajectory(ssm, ys)
        return nothing
    end

    chain = Turing.sample(
        rng,
        joint_drift_model_reg_mooncake(ys),
        Turing.NUTS(N_adapts, 0.8; adtype=AutoMooncake(; config=nothing)),
        N_iter;
        progress=false,
        chain_type=MCMCChains.Chains,
    )

    post_samples = vec(Array(chain[Symbol("b[1]")]))[(N_adapts + 1):end]

    @test mean(post_samples) ≈ kf_post.mean[1] rtol = 1e-1
    @test std(post_samples) ≈ kf_post.std[1] rtol = 2e-1
end

## Joint NUTS: RB SSM with Mooncake ############################################################
# Runs NUTS directly on (b, u₀:T) — no ParticleGibbs alternation. Reverse AD traverses
# conditional-model construction and the shared marginal likelihood evaluator.

@testitem "Joint NUTS: RB SSM with Mooncake" tags = [:mooncake] begin
    using GeneralisedFilters
    using ADTypes: AutoMooncake
    using MCMCChains: MCMCChains
    using Turing
    using StableRNGs
    using Distributions
    using PDMats
    using LinearAlgebra
    using Statistics
    using Mooncake

    rng = StableRNG(42)

    Dx, Dz, Dy = 1, 1, 1
    T_len = 10
    N_iter = 2000
    N_adapts = 500
    σ²_b = 4.0

    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, Dx, Dz, Dy; static_arrays=false
    )
    _, _, ys = simulate(rng, hier_model, T_len)

    drift_indices = (Dx + 1):(Dx + Dz)
    kf_post = GeneralisedFilters.GFTest.augmented_kf_drift_posterior(
        full_model, ys, drift_indices; σ²_b=σ²_b, ε=1e-12
    )

    @model function joint_drift_model_rb_mooncake(ys)
        b ~ MvNormal(zeros(Dz), σ²_b * I)
        ssm = GeneralisedFilters.GFTest.with_inner_drift(hier_model, b)
        x ~ SSMTrajectory(ssm, KF(), ys)
        return nothing
    end

    chain = Turing.sample(
        rng,
        joint_drift_model_rb_mooncake(ys),
        Turing.NUTS(N_adapts, 0.8; adtype=AutoMooncake(; config=nothing)),
        N_iter;
        progress=false,
        chain_type=MCMCChains.Chains,
    )

    post_samples = vec(Array(chain[Symbol("b[1]")]))[(N_adapts + 1):end]

    @test mean(post_samples) ≈ kf_post.mean[1] rtol = 1e-1
    @test std(post_samples) ≈ kf_post.std[1] rtol = 2e-1
end

@testitem "Turing RBPG constrained shared parameters and refreshed targets" tags = [
    :mooncake
] begin
    using Turing, AdvancedHMC, AbstractMCMC, ForwardDiff, Mooncake, ADTypes
    using StaticArrays, Distributions, Random, LogDensityProblems, DifferentiationInterface
    build(b, q) = StateSpaceModel(
        GaussianPrior(SVector(b), SMatrix{1,1}(q)),
        LinearGaussianDynamics(SMatrix{1,1}(0.7), SVector(b), SMatrix{1,1}(q)),
        ctx -> GaussianPrior(SVector(b + 0.2ctx.x0[1]), SMatrix{1,1}(q)),
        ctx -> LinearGaussianDynamics(
            SMatrix{1,1}(0.8), SVector(b + 0.1ctx.x_new[1]), SMatrix{1,1}(q)
        ),
        ctx -> LinearGaussianObservation(
            SMatrix{1,1}(1.0), SVector(0.2ctx.x[1]), SMatrix{1,1}(q)
        ),
    )
    @model function constrained_rb(ys)
        b ~ Normal(0, 1)
        q ~ LogNormal(-1, 0.3)
        return x ~ SSMTrajectory(build(b, q), KF(), ys)
    end
    ys = [SVector(0.1), SVector(0.2)]
    for ad in (AutoForwardDiff(), AutoMooncake(; config=nothing))
        rng = MersenneTwister(92)
        pg = ParticleGibbs(
            ConditionalSMC(RBPF(BF(12; resampler=GeneralisedFilters.Multinomial()), KF())),
            AdvancedHMC.NUTS(0.8);
            adtype=ad,
        )
        _, st = AbstractMCMC.step(rng, constrained_rb(ys), pg; n_adapts=2)
        first_traj = copy(st.trajectory)
        for i in 1:3
            _, st = AbstractMCMC.step(rng, constrained_rb(ys), pg, st; n_adapts=2)
            @test all(isfinite, AbstractMCMC.getparams(st.param_state))
            @test all(x -> x isa AbstractVector, st.trajectory)
            @test st.param_state.i == i + 1
        end
        @test first_traj != st.trajectory
    end
end

@testitem "parameter sampler refreshes changed conditional density" begin
    using AbstractMCMC, AdvancedHMC, ForwardDiff, ADTypes, LogDensityProblems, Distributions
    using Random
    build(θ) = create_homogeneous_linear_gaussian_model(
        [0.0], [1.0;;], [0.8;;], [θ[1]], [0.2;;], [1.0;;], [0.0], [0.3;;]
    )
    pm = ParticleGibbsModel(MvNormal([0.0], [1.0]), ParameterisedSSM(build, [[0.1]]))
    ld1 = GeneralisedFilters._create_log_density_model(
        pm, nothing, [[0.0], [0.1]], AutoForwardDiff()
    )
    ld2 = GeneralisedFilters._create_log_density_model(
        pm, nothing, [[0.0], [1.0]], AutoForwardDiff()
    )
    _, state = AbstractMCMC.step(
        MersenneTwister(1), ld1, AdvancedHMC.NUTS(0.8); initial_params=[0.2], n_adapts=2
    )
    θ = AbstractMCMC.getparams(state)
    refreshed = AbstractMCMC.setparams!!(ld2, state, θ)
    @test refreshed.i == state.i
    @test refreshed.adaptor === state.adaptor
    @test refreshed.transition.z.ℓπ.value ≈ LogDensityProblems.logdensity(ld2.logdensity, θ)
    @test refreshed.transition.z.ℓπ.value != state.transition.z.ℓπ.value
end

@testitem "Turing constrained marginal target includes one Jacobian" tags = [:mooncake] begin
    using Turing, DynamicPPL, LogDensityProblems, ADTypes, ForwardDiff, Mooncake
    using Distributions, Random, StaticArrays, FiniteDifferences
    build(q) = StateSpaceModel(
        GaussianPrior(SVector(q), SMatrix{1,1}(q)),
        LinearGaussianDynamics(SMatrix{1,1}(0.7), SVector(q), SMatrix{1,1}(q)),
        ctx -> GaussianPrior(SVector(q + ctx.x0[1]), SMatrix{1,1}(q)),
        ctx -> LinearGaussianDynamics(
            SMatrix{1,1}(0.8), SVector(q + ctx.x_new[1]), SMatrix{1,1}(q)
        ),
        LinearGaussianObservation(SMatrix{1,1}(1.0), SVector(0.0), SMatrix{1,1}(0.5)),
    )
    ys = [SVector(0.1), SVector(0.2)]
    xs = [SVector(0.1), SVector(0.3), SVector(-0.1)]
    @model function constrained_target(ys)
        q ~ LogNormal(-1, 0.3)
        return x ~ SSMTrajectory(build(q), KF(), ys)
    end
    cm = constrained_target(ys) | (x=reduce(vcat, xs),)
    vi = DynamicPPL.link!!(DynamicPPL.VarInfo(MersenneTwister(1), cm), cm)
    manual(θ) =
        logpdf(LogNormal(-1, 0.3), exp(θ[1])) +
        θ[1] +
        trajectory_logdensity(build(exp(θ[1])), KF(), xs, ys)
    θ = [log(0.4)]
    for ad in (AutoForwardDiff(), AutoMooncake(; config=nothing))
        ld = DynamicPPL.LogDensityFunction(
            cm, DynamicPPL.getlogjoint_internal, vi; adtype=ad
        )
        value, grad = LogDensityProblems.logdensity_and_gradient(ld, θ)
        @test value ≈ manual(θ)
        @test grad ≈ FiniteDifferences.grad(central_fdm(5, 1), manual, θ)[1] rtol = 1e-6
    end
end

@testitem "SSMTrajectory scalar and StaticArray round trips" begin
    using Distributions, StaticArrays, Random
    scalar = StateSpaceModel(
        DistributionPrior(Normal()),
        DistributionDynamics((t, x) -> Normal(0.7x, 0.3)),
        DistributionObservation((t, x) -> Normal(x, 0.5)),
    )
    static = create_homogeneous_linear_gaussian_model(
        SVector(0.0),
        SMatrix{1,1}(1.0),
        SMatrix{1,1}(0.7),
        SVector(0.0),
        SMatrix{1,1}(0.3),
        SMatrix{1,1}(1.0),
        SVector(0.0),
        SMatrix{1,1}(0.5),
    )
    for model in (scalar, static)
        x0, xs, ys = simulate(MersenneTwister(23), model, 3)
        ref = ReferenceTrajectory(x0, xs)
        d = SSMTrajectory(model, ys)
        flat = GeneralisedFilters._flatten_trajectory(ref, 3, 1)
        restored = GeneralisedFilters._trajectory_states(d, flat)
        @test typeof(first(restored)) == typeof(x0)
        @test restored == [ref[t] for t in 0:3]
        @test logpdf(d, flat) ≈ trajectory_logdensity(model, ref, ys)
        @test_throws DimensionMismatch GeneralisedFilters._flatten_trajectory(ref, 2, 1)
        @test_throws DimensionMismatch logpdf(d, flat[1:3])
    end
end

@testitem "SSMTrajectory initialization samples only represented latent states" begin
    using Distributions, StaticArrays, Random
    model = StateSpaceModel(
        GaussianPrior(SVector(0.0), SMatrix{1,1}(1.0)),
        LinearGaussianDynamics(SMatrix{1,1}(0.7), SVector(0.0), SMatrix{1,1}(0.2)),
        GaussianPrior(SVector(0.0), SMatrix{1,1}(0.0)),
        ctx -> LinearGaussianDynamics(
            SMatrix{1,1}(1.0), SVector(ctx.x_new[1]), SMatrix{1,1}(0.0)
        ),
        LinearGaussianObservation(SMatrix{1,1}(1.0), SVector(0.0), SMatrix{1,1}(0.5)),
    )
    d = SSMTrajectory(model, KF(), [SVector(0.1), SVector(0.2)])
    flat = rand(MersenneTwister(81), d)
    @test length(flat) == 3
    @test all(isfinite, flat)
    @test isfinite(logpdf(d, flat))
    @test rand(MersenneTwister(81), d) == flat

    # Observation simulation is optional in the model protocol and must not be required
    # when drawing a trajectory for fixed observations.
    struct DensityOnlyObservation <: ObservationProcess end
    GeneralisedFilters.logdensity(::DensityOnlyObservation, t::Integer, x, y) =
        logpdf(Normal(x, 0.5), y)
    scalar = StateSpaceModel(
        DistributionPrior(Normal()),
        DistributionDynamics((t, x) -> Normal(0.7x, 0.3)),
        DensityOnlyObservation(),
    )
    factor = SSMTrajectory(scalar, [0.1, 0.2])
    values = rand(MersenneTwister(81), factor)
    @test length(values) == 3
    @test isfinite(logpdf(factor, values))
end
