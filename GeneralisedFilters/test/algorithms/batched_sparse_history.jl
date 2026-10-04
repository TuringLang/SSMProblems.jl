@testitem "Sparse GPU CSMC repeated sweeps with and without resampling" tags=[
    :gpu, :batched
] begin
    include(joinpath(@__DIR__, "..", "..", "examples", "gpu-rbpf", "model.jl"))
    using .GPUVolatilityExample, CUDA, BatchedKernels
    const GF = GeneralisedFilters
    CUDA.allowscalar(false)
    cpu, gpu = GPUVolatilityExample.models(2, 1)
    ys = [CuArray(Vector(y)) for y in GPUVolatilityExample.observations(cpu, 5)]
    for strategy in (NoRefreshment(), AncestorSampling()), threshold in (0.0, 1.0)
        algo = ConditionalSMC(RBPF(BF(33; threshold), KF()), strategy)
        reference, ll = GF._csmc_sample(CUDA.RNG(901), gpu, algo, ys, nothing)
        @test isfinite(ll)
        original = [Array(x) for x in collect(reference)]
        second, ll = GF._csmc_sample(CUDA.RNG(902), gpu, algo, ys, reference)
        @test isfinite(ll)
        @test length(second) == 6
        @test all(x -> x isa CUDA.AnyCuVector{Float32}, second)
        @test [Array(x) for x in collect(reference)] == original
    end
end
