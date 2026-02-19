"""Tests for the Turing @model integration with ParticleGibbs."""

## NUTS: smoke test ############################################################################

@testitem "ParticleGibbs Turing NUTS: smoke test" begin
    using GeneralisedFilters
    using AbstractMCMC: AbstractMCMC
    using AdvancedHMC: NUTS
    using MCMCChains: MCMCChains
    using Turing: @model
    using DynamicPPL: DynamicPPL
    using StableRNGs
    using Distributions
    using PDMats
    using LinearAlgebra
    using SSMProblems
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
    _, _, ys = SSMProblems.sample(rng, true_ssm, T_len)

    @model function drift_model_smoke(ys)
        b ~ MvNormal([0.0], 4.0 * I)
        ssm = build_ssm_smoke(b)
        x ~ SSMTrajectory(ssm, ys)
    end

    m = drift_model_smoke(ys)
    pg = ParticleGibbs(CSMC(BF(N_particles)), NUTS(0.8))

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
    using Turing: @model
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
    _, _, ys = SSMProblems.sample(rng, true_ssm, T_len)

    @model function drift_model_mh(ys)
        b ~ MvNormal([0.0], 4.0 * I)
        ssm = build_ssm_mh(b)
        x ~ SSMTrajectory(ssm, ys)
    end

    m = drift_model_mh(ys)
    pg = ParticleGibbs(CSMC(BF(N_particles)), RWMH(MvNormal(zeros(1), 0.5 * I)))

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
    using Turing: @model
    using StableRNGs
    using Distributions
    using PDMats
    using LinearAlgebra
    using Statistics
    using SSMProblems
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
    _, _, ys = SSMProblems.sample(rng, true_ssm, T_len)

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
    end

    m = drift_model_reg(ys)
    pg = ParticleGibbs(CSMC(BF(N_particles)), NUTS(0.8))

    chain = AbstractMCMC.sample(
        rng, m, pg, N_iter; n_adapts=N_adapts, progress=false, chain_type=MCMCChains.Chains
    )

    # Find the parameter column (not trajectory)
    post_samples = Array(chain[:b])[(N_adapts + 1):end]

    @test mean(post_samples) ≈ kf_mean rtol = 0.1
    @test std(post_samples) ≈ kf_std rtol = 0.2
end

## NUTS: HierarchicalSSM against augmented KF ##################################################

@testitem "ParticleGibbs Turing NUTS: HierarchicalSSM" begin
    using GeneralisedFilters
    using AbstractMCMC: AbstractMCMC
    using AdvancedHMC: NUTS
    using MCMCChains: MCMCChains
    using ADTypes: ADTypes
    using Turing: @model
    using StableRNGs
    using Distributions
    using PDMats
    using LinearAlgebra
    using Statistics
    using SSMProblems
    using Zygote

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
    _, _, _, _, ys = SSMProblems.sample(rng, hier_model, T_len)

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
        x ~ SSMTrajectory(ssm, ys)
    end

    m = drift_model_hier(ys)
    pg = ParticleGibbs(
        CSMC(RBPF(BF(N_particles), KF())), NUTS(0.8); adtype=ADTypes.AutoZygote()
    )

    chain = AbstractMCMC.sample(
        rng, m, pg, N_iter; n_adapts=N_adapts, progress=false, chain_type=MCMCChains.Chains
    )

    post_samples = Array(chain[:b])[(N_adapts + 1):end]

    @test mean(post_samples) ≈ kf_mean[1] rtol = 1e-1
    @test std(post_samples) ≈ kf_std[1] rtol = 1e-1
end
