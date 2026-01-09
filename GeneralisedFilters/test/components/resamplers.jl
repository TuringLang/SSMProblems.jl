"""Unit tests for resampling methods."""

## CPU TESTS ###############################################################################

@testsnippet CPUResamplingTestSetup begin
    using Distributions: Normal, pdf
    using Random
    using StableRNGs

    SEED = 1234
    N = 10^6

    rng = StableRNG(SEED)

    xs = rand(rng, Normal(0, 1), N)
    ws = map(x -> pdf(Normal(1, 1), x) / pdf(Normal(0, 1), x), xs)
    ws ./= sum(ws)

    μ0 = sum(ws .* xs)
end

@testitem "CPU multinomial resampling" setup = [CPUResamplingTestSetup] begin
    idxs = GeneralisedFilters.sample_ancestors(rng, Multinomial(), ws)
    @test length(idxs) == N
    μ1 = sum(xs[idxs]) / N
    @test μ0 ≈ μ1 rtol = 1e-1
end

@testitem "CPU systematic resampling" setup = [CPUResamplingTestSetup] begin
    idxs = GeneralisedFilters.sample_ancestors(rng, Systematic(), ws)
    @test length(idxs) == N
    μ1 = sum(xs[idxs]) / N
    @test μ0 ≈ μ1 rtol = 1e-1
end

## GPU TESTS ###############################################################################

@testsnippet GPUResamplingTestSetup begin
    using CUDA
    using Distributions: Normal, pdf
    using Random

    SEED = 1234
    N = 10^6

    rng = CUDA.RNG(SEED)

    xs = randn(rng, N)
    ws = map(x -> pdf(Normal(1, 1), x) / pdf(Normal(0, 1), x), xs)
    ws ./= sum(ws)

    μ0 = sum(ws .* xs)
end

@testitem "GPU multinomial resampling" setup = [GPUResamplingTestSetup] tags = [:gpu] begin
    idxs = GeneralisedFilters.sample_ancestors(rng, Multinomial(), ws)
    @test length(idxs) == N
    μ1 = sum(xs[idxs]) / N
    @test μ0 ≈ μ1 rtol = 1e-1
end

@testitem "GPU systematic resampling" setup = [GPUResamplingTestSetup] tags = [:gpu] begin
    idxs = GeneralisedFilters.sample_ancestors(rng, Systematic(), ws)
    @test length(idxs) == N
    μ1 = sum(xs[idxs]) / N
    @test μ0 ≈ μ1 rtol = 1e-1
end

@testitem "GPU stratified resampling" setup = [GPUResamplingTestSetup] tags = [:gpu] begin
    idxs = GeneralisedFilters.sample_ancestors(rng, Stratified(), ws)
    @test length(idxs) == N
    μ1 = sum(xs[idxs]) / N
    @test μ0 ≈ μ1 rtol = 1e-1
end

@testitem "GPU offspring-to-ancestors" tags = [:gpu] begin
    using CUDA
    offspring = CuVector{Int}([0, 2, 2, 3, 5])
    true_ancestors = CuVector{Int}([2, 2, 4, 5, 5])
    ancestors = GeneralisedFilters.offspring_to_ancestors(offspring)
    @test ancestors == true_ancestors
end

@testitem "GPU ancestors-to-offspring" tags = [:gpu] begin
    using CUDA
    ancestors = CuVector{Int}([4, 2, 2, 3, 1])
    true_offspring = CuVector{Int}([1, 2, 1, 1, 0])
    offspring = GeneralisedFilters.ancestors_to_offspring(ancestors)
    @test offspring == true_offspring
end
