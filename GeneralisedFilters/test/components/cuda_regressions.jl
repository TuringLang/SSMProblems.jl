@testitem "GPU resampling sample counts and passed RNG" tags = [:gpu] begin
    using GeneralisedFilters
    using CUDA
    using Random

    for resampler in (GeneralisedFilters.Multinomial(), Systematic(), Stratified())
        # All mass on the final input catches using W[n] when n differs from K.
        weights = CuArray(Float32[0, 0, 1])
        for n in (0, 1, 3, 7)
            result = GeneralisedFilters.sample_ancestors(
                MersenneTwister(31), resampler, weights, n
            )
            @test length(result) == n
            @test Array(result) == fill(3, n)
        end
        @test isempty(
            GeneralisedFilters.sample_ancestors(
                MersenneTwister(31), resampler, CuArray(Float32[]), 0
            ),
        )
        @test_throws ArgumentError GeneralisedFilters.sample_ancestors(
            MersenneTwister(31), resampler, weights, -1
        )
        @test_throws ArgumentError GeneralisedFilters.sample_ancestors(
            MersenneTwister(31), resampler, CuArray(Float32[]), 1
        )

        weights = CuArray(Float32[0.2, 0.3, 0.5])
        for make_rng in (() -> MersenneTwister(42), () -> CUDA.RNG(42))
            a = GeneralisedFilters.sample_ancestors(make_rng(), resampler, weights, 37)
            b = GeneralisedFilters.sample_ancestors(make_rng(), resampler, weights, 37)
            @test Array(a) == Array(b)
            @test all(i -> 1 <= i <= 3, Array(a))
        end
    end
end

@testitem "GPU cumulative offspring determines ancestor length" tags = [:gpu] begin
    using GeneralisedFilters
    using CUDA
    ext = Base.get_extension(GeneralisedFilters, :CUDAExt)
    @test Array(ext.offspring_to_ancestors(CuArray([0, 0, 2]))) == [3, 3]
    @test Array(ext.offspring_to_ancestors(CuArray([1, 1, 7]))) == [1, 3, 3, 3, 3, 3, 3]
    @test isempty(ext.offspring_to_ancestors(CuArray([0, 0, 0])))
    @test isempty(ext.offspring_to_ancestors(CuArray(Int[])))
    @test isempty(ext.ancestors_to_offspring(CuArray(Int[])))
end

@testitem "GPU tree ancestry uses state element type" tags = [:gpu] begin
    using GeneralisedFilters
    using CUDA
    ext = Base.get_extension(GeneralisedFilters, :CUDAExt)
    tree = ext.ParallelParticleTree(CuArray(Float32[1, 2]), 8)
    initial = ext.get_ancestry(tree, 0)
    @test initial[1].x0 === 1.0f0
    @test eltype(initial[1].xs) == Float32
    @test isempty(ext.get_ancestry(tree, 1, 0).xs)
    ext.insert!(tree, CuArray(Float32[3, 4]), CuArray(Int64[2, 1]))
    paths = ext.get_ancestry(tree, 1)
    @test paths[1].x0 === 2.0f0
    @test paths[1].xs == Float32[3]
    @test paths[2].x0 === 1.0f0
    @test paths[2].xs == Float32[4]
    @test ext.get_ancestry(tree, 1, 1) == paths[1]
end
