"""Tests for threaded evaluation of particle populations."""

@testitem "Threaded population maps" begin
    using GeneralisedFilters
    using StableRNGs
    const GF = GeneralisedFilters

    @test_throws ArgumentError ThreadedExecution(; blocksize=0)
    @test_throws ArgumentError ThreadedExecution(; ntasks=0)

    # Deterministic maps match serial evaluation exactly, including partial final blocks.
    for n in (1, 5, 32, 100), blocksize in (1, 3, 32)
        ex = ThreadedExecution(; blocksize, ntasks=4)
        @test GF._population_map(sin, ex, n) == map(sin, 1:n)
    end

    # Random maps depend on the generator state and blocksize but not on scheduling.
    draws(ex, n) = GF._population_map((rng, i) -> randn(rng), ex, StableRNG(7), n)
    for n in (1, 5, 100), blocksize in (1, 3, 32)
        reference = draws(ThreadedExecution(; blocksize, ntasks=1), n)
        @test allunique(reference)
        for ntasks in (2, 4, nothing)
            @test draws(ThreadedExecution(; blocksize, ntasks), n) == reference
        end
    end

    # The caller's generator advances by exactly one draw.
    rng = StableRNG(3)
    GF._population_map((rng, i) -> rand(rng), ThreadedExecution(; ntasks=2), rng, 100)
    expected = StableRNG(3)
    rand(expected, UInt64)
    @test rand(rng) == rand(expected)

    # Failures keep their own type, whether raised on the calling task or a worker.
    ex = ThreadedExecution(; blocksize=4, ntasks=3)
    for bad in (2, 11)
        @test_throws DomainError GF._population_map(
            i -> i == bad ? throw(DomainError(i)) : i, ex, 20
        )
    end
    @test_throws ArgumentError GF._population_map(i -> i == 7 ? 1.0 : 1, ex, 20)
end

@testitem "Threaded particle filters" begin
    using GeneralisedFilters
    using StableRNGs
    using StatsBase: weights
    const GF = GeneralisedFilters

    rng = StableRNG(1234)
    model = GF.GFTest.create_linear_gaussian_model(rng, 1, 1; static_arrays=true)
    _, _, ys = simulate(rng, model, 4)
    kf_state, llkf = GF.filter(rng, model, KF(), ys)

    full_model, hier_model = GF.GFTest.create_dummy_linear_gaussian_model(
        rng, 1, 1, 1; static_arrays=true
    )
    _, _, hier_ys = simulate(rng, full_model, 4)
    joint_state, ll_joint = GF.filter(rng, full_model, KF(), hier_ys)

    N = 10^5
    prop = GF.GFTest.OptimalProposal(model.dyn, model.obs)
    function filters(ex)
        bf = BF(N; threshold=0.5, execution=ex)
        return (
            bf,
            ParticleFilter(N, prop; threshold=0.5, execution=ex),
            AuxiliaryParticleFilter(bf, MeanPredictive()),
            AuxiliaryParticleFilter(bf, DrawPredictive()),
        )
    end
    function rb_filters(ex)
        rbpf = RBPF(BF(N; threshold=0.5, execution=ex), KF())
        return (rbpf, AuxiliaryParticleFilter(rbpf, MeanPredictive()))
    end
    reference = ThreadedExecution(; ntasks=1)
    schedules = (ThreadedExecution(; ntasks=3), ThreadedExecution())

    for (i, algo) in enumerate(filters(reference))
        state, ll = GF.filter(StableRNG(99), model, algo, ys)
        xs = getfield.(state.particles, :state)
        @test sum(first.(xs) .* weights(state)) ≈ first(kf_state.μ) atol = 0.05
        @test ll ≈ llkf atol = 0.05
        for ex in schedules
            other, other_ll = GF.filter(StableRNG(99), model, filters(ex)[i], ys)
            @test other_ll == ll
            @test other.particles == state.particles
        end
    end

    for (i, algo) in enumerate(rb_filters(reference))
        state, ll = GF.filter(StableRNG(99), hier_model, algo, hier_ys)
        xs = getfield.(getfield.(state.particles, :state), :x)
        zs = getfield.(getfield.(state.particles, :state), :z)
        ws = weights(state)
        @test sum(only.(xs) .* ws) ≈ first(joint_state.μ) atol = 0.05
        @test sum(only.(getfield.(zs, :μ)) .* ws) ≈ last(joint_state.μ) atol = 0.05
        @test ll ≈ ll_joint atol = 0.05
        for ex in schedules
            other, other_ll = GF.filter(
                StableRNG(99), hier_model, rb_filters(ex)[i], hier_ys
            )
            @test other_ll == ll
            @test other.particles == state.particles
        end
    end

    # Observation updates draw no randomness, so threaded and serial updates agree exactly.
    serial = BF(1000)
    threaded = BF(1000; execution=ThreadedExecution(; blocksize=7, ntasks=3))
    initial = GF.initialise(StableRNG(4), model.prior, serial)
    predicted = GF.predict(StableRNG(5), model.dyn, serial, 1, initial, ys[1])
    serial_state, serial_ll = GF.update(model.obs, serial, 1, predicted, ys[1])
    threaded_state, threaded_ll = GF.update(model.obs, threaded, 1, predicted, ys[1])
    @test threaded_ll == serial_ll
    @test threaded_state.particles == serial_state.particles
end
