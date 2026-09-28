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

@testitem "Threaded CSMC refreshment" begin
    using GeneralisedFilters
    using StableRNGs
    using StaticArrays
    using Statistics
    const GF = GeneralisedFilters

    # The joint Gaussian model of the CSMC posterior test and its hierarchical split.
    ax, az, u, v = 0.6, 0.7, 0.35, -0.2
    qx, qz, r = 0.4, 0.25, 0.3
    outer_prior = GaussianPrior(SVector(0.0), SMatrix{1,1}(0.8))
    outer_dyn = TimeVaryingDynamics(
        ((; t),) ->
            LinearGaussianDynamics(SMatrix{1,1}(ax), SVector(0.03t), SMatrix{1,1}(qx)),
    )
    inner_prior_fn = ((; x0),) -> GaussianPrior(SVector(0.3x0[1]), SMatrix{1,1}(0.5))
    inner_dyn_fn =
        ((; t, x_prev, x_new),) -> LinearGaussianDynamics(
            SMatrix{1,1}(az),
            SVector(u * x_prev[1] + v * x_new[1] + 0.02t),
            SMatrix{1,1}(qz),
        )
    inner_obs =
        ((; x),) ->
            LinearGaussianObservation(SMatrix{1,1}(1.0), SVector(0.4x[1]), SMatrix{1,1}(r))
    hier = StateSpaceModel(outer_prior, outer_dyn, inner_prior_fn, inner_dyn_fn, inner_obs)
    joint = StateSpaceModel(
        GaussianPrior(SVector(0.0, 0.0), @SMatrix [0.8 0.24; 0.24 0.572]),
        TimeVaryingDynamics(
            ((; t),) -> LinearGaussianDynamics(
                @SMatrix([ax 0.0; u+v * ax az]),
                SVector(0.03t, (v * 0.03 + 0.02) * t),
                @SMatrix([qx v*qx; v*qx qz+v^2 * qx])
            ),
        ),
        LinearGaussianObservation(@SMatrix([0.4 1.0]), SVector(0.0), SMatrix{1,1}(r)),
    )
    ys = [SVector(0.2), SVector(-0.4), SVector(0.7)]

    function sampler(strategy, rb, ex)
        pf = BF(12; resampler=Multinomial(), threshold=0.8, execution=ex)
        return ConditionalSMC(rb ? RBPF(pf, KF()) : pf, strategy)
    end
    function sweeps(model, csmc, n)
        rng = StableRNG(11)
        ref = nothing
        lls = Float64[]
        for _ in 1:n
            ref, ll = GF._csmc_sample(rng, model, csmc, ys, ref)
            push!(lls, ll)
        end
        return ref, lls
    end

    # Refreshed trajectories do not depend on scheduling.
    for strategy in (AncestorSampling(), BackwardSimulation()), rb in (false, true)
        model = rb ? hier : joint
        reference = sweeps(
            model, sampler(strategy, rb, ThreadedExecution(; blocksize=5, ntasks=1)), 5
        )
        for ntasks in (2, 4)
            ex = ThreadedExecution(; blocksize=5, ntasks)
            @test sweeps(model, sampler(strategy, rb, ex), 5) == reference
        end
    end

    # Threaded Rao–Blackwellised refreshment targets the joint Gaussian posterior.
    rng = StableRNG(8251)
    truth, _ = smooth(rng, joint, KalmanSmoother(), ys; t_smooth=1)
    for strategy in (AncestorSampling(), BackwardSimulation())
        csmc = sampler(strategy, true, ThreadedExecution(; blocksize=4))
        draws = Float64[]
        ref = nothing
        for i in 1:4200
            ref, _ = GF._csmc_sample(rng, hier, csmc, ys, ref)
            i > 200 && push!(draws, only(ref[1]))
        end
        @test mean(draws) ≈ truth.μ[1] atol = 0.055
        @test mean(abs2, draws) ≈ truth.Σ[1, 1] + truth.μ[1]^2 atol = 0.07
    end
end
