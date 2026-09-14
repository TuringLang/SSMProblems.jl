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

@testitem "CPU stratified resampling" setup = [CPUResamplingTestSetup] begin
    idxs = GeneralisedFilters.sample_ancestors(rng, Stratified(), ws)
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
    ext = Base.get_extension(GeneralisedFilters, :CUDAExt)
    offspring = CuVector{Int}([0, 2, 2, 3, 5])
    true_ancestors = CuVector{Int}([2, 2, 4, 5, 5])
    ancestors = ext.offspring_to_ancestors(offspring)
    @test ancestors == true_ancestors
end

@testitem "GPU ancestors-to-offspring" tags = [:gpu] begin
    using CUDA
    ext = Base.get_extension(GeneralisedFilters, :CUDAExt)
    ancestors = CuVector{Int}([4, 2, 2, 3, 1])
    true_offspring = CuVector{Int}([1, 2, 1, 1, 0])
    offspring = ext.ancestors_to_offspring(ancestors)
    @test offspring == true_offspring
end

## CONDITIONAL RESAMPLING ##################################################################

@testsnippet ConditionalResamplingTestSetup begin
    using GeneralisedFilters
    using StableRNGs

    """
        mean_offspring(rng, draw, weights, reps)

    Monte Carlo estimate of `sum_a W[a] * E[offspring counts | reference = a] / M`, the
    left-hand side of the invariance condition (Definition 9 of Finke, Johansen, Lee &
    Murray, arXiv:2606.25603) evaluated at the indicator of each particle.

    Indicators span the symmetric linear functionals of the resampled empirical measure, so
    a scheme is invariant exactly when this equals `weights`.
    """
    function mean_offspring(rng, draw, weights, reps)
        n = length(weights)
        total = zeros(Float64, n)
        for ref_idx in 1:n
            counts = zeros(Float64, n)
            for _ in 1:reps
                idxs = draw(rng, weights, ref_idx)
                @assert idxs[1] == ref_idx
                for idx in idxs
                    counts[idx] += 1
                end
            end
            total .+= weights[ref_idx] .* counts ./ (reps * n)
        end
        return total
    end

    conditional_draw(resampler) =
        (rng, weights, ref_idx) -> GeneralisedFilters.conditional_sample_ancestors(
            rng, resampler, weights, ref_idx
        )

    # The shortcut this interface replaces: draw unconditionally, then overwrite the first
    # index (Strategy I of Section 5.3 of the reference).
    naive_draw(resampler) =
        (rng, weights, ref_idx) -> begin
            idxs = GeneralisedFilters.sample_ancestors(rng, resampler, weights)
            idxs[1] = ref_idx
            return idxs
        end

    RESAMPLERS = (Multinomial(), Systematic(), Stratified())

    # Shared by the invariance test and its negative control, so that loosening the
    # tolerance to admit an invalid scheme also makes the control fail.
    INVARIANCE_WEIGHTS = [0.6, 0.1, 0.1, 0.1, 0.1]
    INVARIANCE_REPS = 100_000
    INVARIANCE_ATOL = 0.01
end

@testitem "Conditional resampling retains the reference" setup = [
    ConditionalResamplingTestSetup
] begin
    rng = StableRNG(1234)
    ws = [0.05, 0.5, 0.3, 0.1, 0.05]

    for resampler in RESAMPLERS, ref_idx in 1:length(ws)
        idxs = GeneralisedFilters.conditional_sample_ancestors(rng, resampler, ws, ref_idx)
        @test length(idxs) == length(ws)
        @test idxs[1] == ref_idx
        @test all(∈(1:length(ws)), idxs)
    end
end

@testitem "Conditional resampling is invariant" setup = [ConditionalResamplingTestSetup] begin
    for resampler in RESAMPLERS
        rng = StableRNG(1234)
        estimate = mean_offspring(
            rng, conditional_draw(resampler), INVARIANCE_WEIGHTS, INVARIANCE_REPS
        )
        @test estimate ≈ INVARIANCE_WEIGHTS atol = INVARIANCE_ATOL
    end
end

# A power check on the invariance test above rather than a test of package code: it holds
# the tolerance there at a value that can distinguish a valid conditional law from an
# invalid one.
@testitem "Overwriting an index breaks invariance" setup = [ConditionalResamplingTestSetup] begin
    # Multinomial draws are independent, so the shortcut coincides with its conditional law.
    rng = StableRNG(1234)
    estimate = mean_offspring(
        rng, naive_draw(Multinomial()), INVARIANCE_WEIGHTS, INVARIANCE_REPS
    )
    @test estimate ≈ INVARIANCE_WEIGHTS atol = INVARIANCE_ATOL

    # The other schemes are not, and the shortcut loses invariance. Systematic and
    # stratified resampling both retain the heaviest particle in the first slot here, so
    # overwriting that slot removes an offspring it was entitled to.
    for resampler in (Systematic(), Stratified())
        local rng = StableRNG(1234)
        local estimate = mean_offspring(
            rng, naive_draw(resampler), INVARIANCE_WEIGHTS, INVARIANCE_REPS
        )
        @test !isapprox(estimate, INVARIANCE_WEIGHTS; atol=INVARIANCE_ATOL)
    end
end

@testitem "Conditional resampling support is declared" begin
    using Random

    @test GeneralisedFilters.supports_conditional(Multinomial())
    @test GeneralisedFilters.supports_conditional(Systematic())
    @test GeneralisedFilters.supports_conditional(Stratified())
    @test !GeneralisedFilters.supports_conditional(Metropolis())
    @test !GeneralisedFilters.supports_conditional(Rejection())

    # Wrappers forward the trait to the scheme they delegate to.
    @test GeneralisedFilters.supports_conditional(
        GeneralisedFilters.ESSResampler(0.5, Systematic())
    )
    @test !GeneralisedFilters.supports_conditional(
        GeneralisedFilters.ESSResampler(0.5, Metropolis())
    )

    @test_throws ArgumentError GeneralisedFilters.conditional_sample_ancestors(
        Random.default_rng(), Metropolis(), [0.5, 0.5], 1
    )
end

@testitem "GPU conditional resampling retains the reference" tags = [:gpu] begin
    using CUDA

    ws = CuVector([0.05, 0.5, 0.3, 0.1, 0.05])
    rng = CUDA.RNG(1234)

    for resampler in (Multinomial(), Systematic(), Stratified()), ref_idx in 1:5
        idxs = GeneralisedFilters.conditional_sample_ancestors(rng, resampler, ws, ref_idx)
        @test length(idxs) == 5
        @test CUDA.@allowscalar idxs[1] == ref_idx
    end
end

@testitem "GPU conditional resampling matches the CPU law" tags = [:gpu] begin
    using CUDA
    using StableRNGs

    # Compare offspring distributions rather than index order: the GPU schemes order slots
    # by ancestor whereas the CPU schemes order them by stratum.
    ws = [0.6, 0.1, 0.1, 0.1, 0.1]
    ws_gpu = CuVector(ws)
    reps = 20_000

    for resampler in (Systematic(), Stratified())
        gpu_rng = CUDA.RNG(1234)
        cpu_rng = StableRNG(1234)
        gpu_counts = zeros(Float64, 5)
        cpu_counts = zeros(Float64, 5)
        for _ in 1:reps
            for idx in Vector(
                GeneralisedFilters.conditional_sample_ancestors(
                    gpu_rng, resampler, ws_gpu, 1
                ),
            )
                gpu_counts[idx] += 1
            end
            for idx in
                GeneralisedFilters.conditional_sample_ancestors(cpu_rng, resampler, ws, 1)
                cpu_counts[idx] += 1
            end
        end
        @test gpu_counts ./ reps ≈ cpu_counts ./ reps atol = 0.05
    end
end

@testitem "GPU offspring-to-ancestors with a cyclical shift" tags = [:gpu] begin
    using CUDA
    ext = Base.get_extension(GeneralisedFilters, :CUDAExt)
    offspring = CuVector{Int}([0, 2, 2, 3, 5])
    # Unshifted: [2, 2, 4, 5, 5].
    @test ext.offspring_to_ancestors(offspring; shift=2) == CuVector{Int}([4, 5, 5, 2, 2])
    @test ext.offspring_to_ancestors(offspring; shift=0) ==
        ext.offspring_to_ancestors(offspring)
end

@testitem "Selected ancestor conditions all offspring and auxiliary corrections" begin
    using GeneralisedFilters, StableRNGs
    const GF = GeneralisedFilters
    weights = [0.02, 0.81, 0.04, 0.13]
    particles = [GF.Particle(i, log(weights[i]), i) for i in eachindex(weights)]
    state = GF.ParticleDistribution(particles, 0.0)
    reference = [1]
    for scheme in (Multinomial(), Systematic(), Stratified()),
        ancestor in eachindex(weights)

        rng = StableRNG(91)
        expected = GF.conditional_sample_ancestors(rng, scheme, weights, ancestor)
        actual = GF.resample(
            StableRNG(91),
            ESSResampler(1.0, scheme),
            state,
            weights;
            ref_state=reference,
            ref_idx=ancestor,
        )
        @test [p.ancestor for p in actual.particles] == expected
        @test [p.state for p in actual.particles] == expected
        @test all(p -> p.log_w == 0, actual.particles)
    end
    kept = GF.maybe_resample(StableRNG(91), ESSResampler(0.0), state; ref_state=reference)
    @test [p.ancestor for p in kept.particles] == collect(eachindex(weights))
    @test [p.log_w for p in kept.particles] == log.(weights)
end
