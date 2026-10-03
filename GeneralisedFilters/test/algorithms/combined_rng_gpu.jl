@testitem "CombinedRNG routes GPU draws and replays complete filtering" tags = [:gpu, :batched] begin
    using CUDA, BatchedKernels, Random, AbstractMCMC
    const GF = GeneralisedFilters
    CUDA.allowscalar(false)

    # The same bundle must preserve each child's native stream independently.
    for T in (Float32, Float64), make_gpu in (BatchedRNG, CUDA.RNG)
        rng = CombinedRNG(Xoshiro(11), make_gpu(12))
        cpu = Xoshiro(11)
        gpu = make_gpu(12)
        @test rand(rng, T, 13) == rand(cpu, T, 13)
        for draw! in (rand!, randn!)
            actual, expected = CuArray{T}(undef, 3, 7), CuArray{T}(undef, 3, 7)
            @test draw!(rng, actual) === actual
            draw!(gpu, expected)
            @test Array(actual) == Array(expected)
        end
        @test rand(rng, UInt64) == rand(cpu, UInt64)
    end

    rng = CombinedRNG(Xoshiro(15), BatchedRNG(16))
    other = CombinedRNG(Xoshiro(17), BatchedRNG(18))
    scratch = CuArray{Float32}(undef, 17)
    rand!(rng, scratch)
    rand(rng)
    Random.seed!(rng, 1234)
    Random.seed!(other, 1234)
    @test gpu_rng(rng).counter == 0
    @test rand(rng, UInt64) == rand(other, UInt64)
    expected = similar(scratch)
    randn!(rng, scratch)
    randn!(other, expected)
    @test Array(scratch) == Array(expected)

    before_view = copy(rng)
    unsupported = view(CUDA.zeros(Float32, 10), 1:2:9)
    @test_throws MethodError rand!(rng, unsupported)
    @test_throws MethodError randn!(rng, unsupported)
    @test gpu_rng(rng).counter == gpu_rng(before_view).counter
    @test rand(rng, UInt64) == rand(before_view, UInt64)

    # Exercise the public filter, including fused initialisation/propagation,
    # bulk device resampling and host scalar resampling with one RNG argument.
    struct BundleObservation end
    function GF.inner_observation(::BundleObservation, ::Integer, x::BatchedCuVector)
        n = length(x)
        return BatchedStruct(LinearGaussianObservation, (;
            H=SharedCuMatrix(CUDA.ones(Float32, 1, 1), n), c=x,
            R=SharedCuMatrix(CUDA.fill(0.2f0, 1, 1), n),
        ))
    end
    model = StateSpaceModel(
        GaussianPrior(CUDA.zeros(Float32, 1), CUDA.ones(Float32, 1, 1)),
        LinearGaussianDynamics(
            CUDA.fill(0.8f0, 1, 1), CUDA.zeros(Float32, 1),
            CovarianceFactor(CUDA.fill(0.2f0, 1, 1)),
        ),
        GaussianPrior(CUDA.zeros(Float32, 1), CUDA.ones(Float32, 1, 1)),
        LinearGaussianDynamics(
            CUDA.fill(0.7f0, 1, 1), CUDA.zeros(Float32, 1), CUDA.fill(0.1f0, 1, 1),
        ),
        BundleObservation(),
    )
    ys = [CuArray(Float32[y]) for y in (0.2, -0.1, 0.3)]
    snapshot(state) = (
        Array(state.particles.components.state.components.x.data),
        Array(state.particles.components.state.components.z.components.μ.data),
        Array(state.particles.components.state.components.z.components.Σ.data),
        Array(state.particles.components.ancestor.data),
        Array(GF.log_weights(state)),
    )
    for scheme in (Multinomial(), Systematic(), Stratified())
        weights = CuArray(Float32[0.1, 0.2, 0.3, 0.4])
        draw_rng = CombinedRNG(Xoshiro(19), BatchedRNG(20))
        draw_replay = copy(draw_rng)
        ancestors = GF.sample_ancestors(draw_rng, scheme, weights, 31)
        @test Array(ancestors) == Array(GF.sample_ancestors(draw_replay, scheme, weights, 31))
        @test all(i -> 1 <= i <= 4, Array(ancestors))
        conditional = GF.conditional_sample_ancestors(draw_rng, scheme, weights, 3)
        @test Array(conditional) == Array(GF.conditional_sample_ancestors(draw_replay, scheme, weights, 3))
        @test Array(conditional)[1] == 3
        algo = RBPF(BF(31; threshold=1.0, resampler=scheme), KF())
        local rng = CombinedRNG(Xoshiro(21), BatchedRNG(22))
        replay = copy(rng)
        first, ll = GF.filter(rng, model, algo, ys)
        second, ll_replay = GF.filter(replay, model, algo, ys)
        @test snapshot(first) == snapshot(second)
        @test ll == ll_replay
        @test isfinite(ll)
        @test rand(rng, UInt64) == rand(replay, UInt64)
        # Continued calls must advance, and copying the advanced bundle replays.
        continuation = copy(rng)
        third, ll_third = GF.filter(rng, model, algo, ys)
        fourth, ll_fourth = GF.filter(continuation, model, algo, ys)
        @test snapshot(third) == snapshot(fourth)
        @test ll_third == ll_fourth
        @test snapshot(first)[1] != snapshot(third)[1]
        @test snapshot(first)[4] != collect(1:31)

        # NoRefreshment uses conditional filtering plus host-controlled selection
        # of a terminal trajectory, all through the same positional RNG.
        sampler = ConditionalSMC(algo)
        csmc_model = CSMCModel(model, ys)
        chain_rng = CombinedRNG(Xoshiro(41), BatchedRNG(42))
        chain_replay = copy(chain_rng)
        draw, state = AbstractMCMC.step(chain_rng, csmc_model, sampler)
        replay_draw, replay_state = AbstractMCMC.step(chain_replay, csmc_model, sampler)
        trajectory(draw) = [Array(x) for x in draw.trajectory]
        @test trajectory(draw) == trajectory(replay_draw)
        draw, state = AbstractMCMC.step(chain_rng, csmc_model, sampler, state)
        replay_draw, replay_state = AbstractMCMC.step(chain_replay, csmc_model, sampler, replay_state)
        @test trajectory(draw) == trajectory(replay_draw)
    end
end
