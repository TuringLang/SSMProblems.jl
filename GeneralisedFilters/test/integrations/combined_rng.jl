@testitem "CombinedRNG supports particle Gibbs with NUTS" begin
    using Random, Distributions, ForwardDiff
    using AbstractMCMC: AbstractMCMC
    using AdvancedHMC: NUTS

    function build_model(θ)
        return create_homogeneous_linear_gaussian_model(
            [0.0], [1.0;;], [0.8;;], [θ[1]], [0.1;;],
            [1.0;;], [0.0], [0.5;;],
        )
    end
    _, _, ys = simulate(Xoshiro(42), build_model([0.2]), 5)
    model = ParticleGibbsModel(
        MvNormal([0.0], [4.0;;]), ParameterisedSSM(build_model, ys)
    )
    sampler = ParticleGibbs(
        ConditionalSMC(BF(10; resampler=GeneralisedFilters.Multinomial())), NUTS(0.8)
    )
    # A CPU stand-in for the device member keeps this integration test GPU-free.
    rng = CombinedRNG(Xoshiro(1234), Xoshiro(5678))
    replay = copy(rng)
    device_before = copy(gpu_rng(rng))
    first, state = AbstractMCMC.step(rng, model, sampler; n_adapts=5)
    expected, expected_state = AbstractMCMC.step(replay, model, sampler; n_adapts=5)
    @test first.θ == expected.θ
    @test all(isfinite, first.θ)
    @test haskey(first.stat, :acceptance_rate)
    second, state = AbstractMCMC.step(rng, model, sampler, state; n_adapts=5)
    expected_second, _ = AbstractMCMC.step(
        replay, model, sampler, expected_state; n_adapts=5
    )
    @test second.θ == expected_second.θ
    @test all(isfinite, second.θ)
    @test rand(gpu_rng(rng), UInt64, 8) == rand(device_before, UInt64, 8)
end

@testitem "CombinedRNG follows AbstractMCMC ensemble copy and seed lifecycle" begin
    using Random
    using AbstractMCMC: AbstractMCMC

    struct BundleProbeModel <: AbstractMCMC.AbstractModel end
    struct BundleProbeSampler <: AbstractMCMC.AbstractSampler end
    function AbstractMCMC.step(
        rng::AbstractRNG, ::BundleProbeModel, ::BundleProbeSampler, state=nothing; kwargs...
    )
        sample = (rand(rng, UInt64), rand(gpu_rng(rng), UInt64))
        return sample, nothing
    end

    fresh() = CombinedRNG(Xoshiro(71), Xoshiro(93))
    serial = AbstractMCMC.sample(
        fresh(), BundleProbeModel(), BundleProbeSampler(), AbstractMCMC.MCMCSerial(),
        4, 3; progress=false,
    )
    replay = AbstractMCMC.sample(
        fresh(), BundleProbeModel(), BundleProbeSampler(), AbstractMCMC.MCMCSerial(),
        4, 3; progress=false,
    )
    @test serial == replay
    @test length(unique(first(chain)[1] for chain in serial)) == 3
    @test length(unique(first(chain)[2] for chain in serial)) == 3
    if Threads.nthreads() > 1
        threaded = AbstractMCMC.sample(
            fresh(), BundleProbeModel(), BundleProbeSampler(), AbstractMCMC.MCMCThreads(),
            4, 3; progress=false,
        )
        @test threaded == serial
    end
end
