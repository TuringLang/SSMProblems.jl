@testitem "CombinedRNG CPU sampling and delegation" begin
    using Random, Distributions, StableRNGs, LinearAlgebra

    for constructor in (Xoshiro, MersenneTwister, StableRNG)
        reference = constructor(123)
        device = Xoshiro(456)
        device_reference = copy(device)
        bundle = CombinedRNG(copy(reference), device)
        @test cpu_rng(bundle) === bundle.cpu
        @test gpu_rng(bundle) === device
        @test cpu_rng(reference) === reference
        @test gpu_rng(reference) === reference

        # Scalar samplers, optimized dense bulk paths, and distribution clients.
        for T in (Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64,
                  Int128, UInt128, Float16, Float32, Float64)
            @test rand(bundle, T) == rand(reference, T)
        end
        @test rand(bundle) == rand(reference)
        @test rand(bundle, 2:29) == rand(reference, 2:29)
        @test rand(bundle, [2, 5, 8]) == rand(reference, [2, 5, 8])
        for T in (Float16, Float32, Float64)
            @test randn(bundle, T) == randn(reference, T)
            @test randexp(bundle, T) == randexp(reference, T)
            @test rand(bundle, T, 8, 17) == rand(reference, T, 8, 17)
            @test randn(bundle, T, 8, 17) == randn(reference, T, 8, 17)
            @test randexp(bundle, T, 8, 17) == randexp(reference, T, 8, 17)
        end
        @test randn(bundle, ComplexF64) == randn(reference, ComplexF64)
        for f in (rand!, randn!, randexp!)
            actual, expected = zeros(128), zeros(128)
            @test f(bundle, actual) === actual
            f(reference, expected)
            @test actual == expected
        end
        for distribution in (Normal(), Gamma(2), Categorical([0.2, 0.8]),
                             MvNormal([0.0, 0.0], [1.0 0.2; 0.2 1.0]),
                             Dirichlet([1.0, 2.0]))
            @test rand(bundle, distribution) == rand(reference, distribution)
            @test rand(bundle, distribution, 10) == rand(reference, distribution, 10)
        end
        # Generic non-Array containers still have working CPU sampler fallbacks.
        @test all(x -> 0 <= x < 1, rand!(bundle, view(zeros(20), 1:2:19)))
        @test length(rand(bundle, Bool, 10)) == 10
        @test rand(device, UInt64) == rand(device_reference, UInt64)
    end
end

@testitem "CombinedRNG lifecycle and ownership" begin
    using Random, StableRNGs

    for constructor in (Xoshiro, MersenneTwister, StableRNG)
        cpu, device = constructor(123), Xoshiro(456)
        bundle = CombinedRNG(cpu, device)
        @test cpu_rng(bundle) === cpu
        @test gpu_rng(bundle) === device
        snapshot = copy(bundle)
        @test snapshot.cpu !== cpu
        @test snapshot.gpu !== device
        @test rand(bundle, UInt64) == rand(snapshot, UInt64)
        @test rand(bundle.gpu, UInt64) == rand(snapshot.gpu, UInt64)
        rand(bundle, UInt64)
        rand(bundle.gpu, UInt64)
        untouched = copy(snapshot)
        @test rand(snapshot, UInt64) == rand(untouched, UInt64)
        @test rand(snapshot.gpu, UInt64) == rand(untouched.gpu, UInt64)

        @test Random.seed!(bundle, 789) === bundle
        @test bundle.cpu === cpu
        @test bundle.gpu === device
        @test rand(bundle, 100) == rand(constructor(789), 100)
        derived_seed = rand(constructor(789), UInt64)
        @test rand(bundle.gpu, 100) == rand(Xoshiro(derived_seed), 100)
        Random.seed!(bundle, 789)
        Random.seed!(snapshot, 789)
        @test rand(bundle, 100) == rand(snapshot, 100)
        @test rand(bundle.gpu, 100) == rand(snapshot.gpu, 100)
    end
    cpu = Xoshiro(1)
    @test_throws ArgumentError CombinedRNG(cpu, cpu)
    @test_throws ArgumentError CombinedRNG(Random.default_rng(), Xoshiro(2))
    @test_throws ArgumentError CombinedRNG(Xoshiro(1), Random.default_rng())
    bundle = CombinedRNG(Xoshiro(1), Xoshiro(2))
    @test_throws ArgumentError CombinedRNG(bundle, Xoshiro(3))
    @test_throws ArgumentError CombinedRNG(Xoshiro(3), bundle)
end

@testitem "CombinedRNG method ambiguities" begin
    using Random, Distributions, StableRNGs

    pairs = Test.detect_ambiguities(GeneralisedFilters, Random, Distributions)
    relevant = Base.filter(pairs) do pair
        any(method -> occursin("CombinedRNG", string(method.sig)), pair)
    end
    @test isempty(relevant)
end
