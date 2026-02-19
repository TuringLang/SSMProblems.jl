"""Tests for the ParticleGibbs sampler."""

## NUTS: smoke test ############################################################################

@testitem "ParticleGibbs NUTS: smoke test" begin
    using GeneralisedFilters
    using AbstractMCMC: AbstractMCMC
    using AdvancedHMC: NUTS
    using MCMCChains: MCMCChains
    using StableRNGs
    using Distributions
    using PDMats
    using LinearAlgebra
    using SSMProblems
    using ForwardDiff

    rng = StableRNG(1234)

    # Simple 1D model: x_t = a*x_{t-1} + b + noise, y_t = x_t + noise
    a = 0.8
    q² = 0.1
    r² = 0.5
    σ₀² = 1.0

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

    true_ssm = build_ssm([1.0])
    _, _, ys = SSMProblems.sample(rng, true_ssm, 5)

    prior = MvNormal([0.0], [4.0;;])
    pssm = ParameterisedSSM(build_ssm, ys)
    model = ParticleGibbsModel(prior, pssm)
    pg = ParticleGibbs(CSMC(BF(10)), NUTS(0.8))

    # Initial step
    transition, state = AbstractMCMC.step(rng, model, pg; n_adapts=5)
    @test transition isa GeneralisedFilters.ParticleGibbsTransition
    @test state isa GeneralisedFilters.ParticleGibbsState
    @test length(transition.θ) == 1
    @test haskey(transition.stat, :acceptance_rate)
    @test haskey(transition.stat, :step_size)
    @test haskey(transition.stat, :tree_depth)

    # Subsequent step
    transition2, state2 = AbstractMCMC.step(rng, model, pg, state; n_adapts=5)
    @test transition2 isa GeneralisedFilters.ParticleGibbsTransition
    @test state2 isa GeneralisedFilters.ParticleGibbsState

    # Chain output via sample
    chain = AbstractMCMC.sample(
        rng, model, pg, 20; n_adapts=10, progress=false, chain_type=MCMCChains.Chains
    )
    @test chain isa MCMCChains.Chains
    @test size(chain, 1) == 20
    @test length(names(chain, :parameters)) == 1

    # Custom parameter names
    chain_named = AbstractMCMC.sample(
        rng,
        model,
        pg,
        20;
        n_adapts=10,
        progress=false,
        chain_type=MCMCChains.Chains,
        param_names=["b"],
    )
    @test :b in names(chain_named, :parameters)
end

## MH: smoke test #############################################################################

@testitem "ParticleGibbs MH: smoke test" begin
    using GeneralisedFilters
    using AbstractMCMC: AbstractMCMC
    using AdvancedMH: RWMH
    using MCMCChains: MCMCChains
    using StableRNGs
    using Distributions
    using PDMats
    using LinearAlgebra
    using SSMProblems

    rng = StableRNG(1234)

    a = 0.8
    q² = 0.1
    r² = 0.5
    σ₀² = 1.0

    function build_ssm_mh(θ)
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

    true_ssm = build_ssm_mh([1.0])
    _, _, ys = SSMProblems.sample(rng, true_ssm, 5)

    prior = MvNormal([0.0], [4.0;;])
    pssm = ParameterisedSSM(build_ssm_mh, ys)
    model = ParticleGibbsModel(prior, pssm)
    pg = ParticleGibbs(CSMC(BF(10)), RWMH(MvNormal(zeros(1), 0.5 * I)))

    # Initial step
    transition, state = AbstractMCMC.step(rng, model, pg)
    @test transition isa GeneralisedFilters.ParticleGibbsTransition
    @test state isa GeneralisedFilters.ParticleGibbsState
    @test length(transition.θ) == 1
    @test haskey(transition.stat, :accepted)

    # Subsequent step
    transition2, state2 = AbstractMCMC.step(rng, model, pg, state)
    @test transition2 isa GeneralisedFilters.ParticleGibbsTransition

    # Chain output
    chain = AbstractMCMC.sample(
        rng, model, pg, 20; progress=false, chain_type=MCMCChains.Chains
    )
    @test chain isa MCMCChains.Chains
    @test size(chain, 1) == 20
    @test :accepted in names(chain, :internals)
end

## NUTS: HierarchicalSSM against augmented KF ###################################################

@testitem "ParticleGibbs NUTS: HierarchicalSSM" begin
    using GeneralisedFilters
    using AbstractMCMC: AbstractMCMC
    using AdvancedHMC: NUTS
    using MCMCChains: MCMCChains
    using ADTypes: ADTypes
    using StableRNGs
    using Distributions
    using PDMats
    using LinearAlgebra
    using Statistics
    using SSMProblems
    using Zygote

    rng = StableRNG(42)

    # Dimensions
    Dx, Dz, Dy = 1, 1, 1
    T_len = 10
    N_particles = 50
    N_iter = 5000
    N_adapts = 500
    σ²_b = 4.0

    # Generate a random hierarchical model and fix everything except the inner drift b
    full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, Dx, Dz, Dy; static_arrays=true
    )
    _, _, _, _, ys = SSMProblems.sample(rng, hier_model, T_len)

    # Parameterise: θ controls inner dynamics drift b
    fixed = hier_model
    build_hier(θ) = GeneralisedFilters.GFTest.with_inner_drift(fixed, θ)

    # Augmented KF ground truth using full linear Gaussian model with unknown inner drift
    drift_indices = (Dx + 1):(Dx + Dz)
    kf_post = GeneralisedFilters.GFTest.augmented_kf_drift_posterior(
        full_model, ys, drift_indices; σ²_b=σ²_b, ε=1e-12
    )
    kf_mean = kf_post.mean
    kf_std = kf_post.std

    # Particle Gibbs with RBPF
    prior = MvNormal(zeros(Dz), σ²_b * I)
    pssm = ParameterisedSSM(build_hier, ys)
    model = ParticleGibbsModel(prior, pssm)
    pg = ParticleGibbs(
        CSMC(RBPF(BF(N_particles), KF())), NUTS(0.8); adtype=ADTypes.AutoZygote()
    )

    chain = AbstractMCMC.sample(
        rng,
        model,
        pg,
        N_iter;
        n_adapts=N_adapts,
        progress=false,
        chain_type=MCMCChains.Chains,
        param_names=["b_z"],
    )

    post_samples = Array(chain[:b_z])[(N_adapts + 1):end]

    @test mean(post_samples) ≈ kf_mean[1] rtol = 1e-1
    @test std(post_samples) ≈ kf_std[1] rtol = 1e-1
end

## NUTS: regular SSM against augmented KF ######################################################

@testitem "ParticleGibbs NUTS: regular SSM" begin
    using GeneralisedFilters
    using AbstractMCMC: AbstractMCMC
    using AdvancedHMC: NUTS
    using MCMCChains: MCMCChains
    using StableRNGs
    using Distributions
    using PDMats
    using LinearAlgebra
    using Statistics
    using SSMProblems
    using ForwardDiff

    rng = StableRNG(42)

    # Model parameters
    a = 0.8
    q² = 0.1
    r² = 0.5
    σ₀² = 1.0
    σ_b² = 4.0
    T_len = 10
    N_particles = 50
    N_iter = 5000
    N_adapts = 500

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

    # Generate data
    true_b = 1.5
    true_ssm = build_ssm([true_b])
    _, _, ys = SSMProblems.sample(rng, true_ssm, T_len)

    # Augmented KF ground truth
    ref_model = build_ssm([0.0])
    kf_post = GeneralisedFilters.GFTest.augmented_kf_drift_posterior(
        ref_model, ys, 1; σ²_b=σ_b², ε=1e-12
    )
    kf_mean = kf_post.mean[1]
    kf_std = kf_post.std[1]

    # Particle Gibbs
    prior = MvNormal([0.0], [σ_b²;;])
    pssm = ParameterisedSSM(build_ssm, ys)
    model = ParticleGibbsModel(prior, pssm)
    pg = ParticleGibbs(CSMC(BF(N_particles)), NUTS(0.8))

    chain = AbstractMCMC.sample(
        rng,
        model,
        pg,
        N_iter;
        n_adapts=N_adapts,
        progress=false,
        chain_type=MCMCChains.Chains,
        param_names=["b"],
    )

    # Compare posterior statistics (discard warmup)
    post_samples = Array(chain[Symbol("b")])[(N_adapts + 1):end]

    @test mean(post_samples) ≈ kf_mean rtol = 0.1
    @test std(post_samples) ≈ kf_std rtol = 0.2
end
