@testitem "RBPF population adapters preserve CPU execution" begin
    using Random, StaticArrays
    const GF = GeneralisedFilters
    _, model = GF.GFTest.create_dummy_linear_gaussian_model(
        Xoshiro(8), 1, 1, 1; static_arrays=true
    )
    n = 7
    for execution in (SerialExecution(), ThreadedExecution(blocksize=2, ntasks=2))
        algo = RBPF(BF(n; execution), KF())
        initial = initialise(Xoshiro(12), model.prior, algo)
        old = [
            GF.Particle(p.state, p.log_w, Int64(2)^40 + i) for
            (i, p) in enumerate(initial.particles)
        ]
        fields = GF._rb_population_fields(old)
        @test fields.state.x == map(p -> p.state.x, old)
        @test fields.ancestor == map(p -> p.ancestor, old)
        @test eltype(fields.ancestor) === typeof(old[1].ancestor)
        assembled = GF._assemble_rb_population(old, fields)
        @test assembled == old
        @test assembled !== old
        @test assembled[1].state.z === old[1].state.z
        @test GF._assemble_rb_population(old, assembled) === assembled
        @test_throws DimensionMismatch GF._assemble_rb_population(
            old, merge(fields, (; ancestor=Int[]))
        )
        # Extracting a CPU field is not a writable projection into the parent.
        original_x = old[1].state.x
        fields.state.x[1] = zero(original_x)
        @test old[1].state.x == original_x

        reference = ReferenceTrajectory(SVector(0.1), [SVector(-0.4)])
        y = SVector(0.2)
        rng, reference_rng = Xoshiro(33), Xoshiro(33)
        actual = GF._predict_particles(rng, model.dyn, algo, 1, old, y, reference)
        expected = GF._population_map(execution, reference_rng, n) do r, i
            GF.predict_particle(
                r, model.dyn, algo, 1, old[i], y, i == 1 ? reference[1] : nothing
            )
        end
        @test rand(rng) == rand(reference_rng)
        for (a, e) in zip(actual, expected)
            @test a.state.x == e.state.x
            @test a.state.z.μ == e.state.z.μ
            @test a.state.z.Σ == e.state.z.Σ
            @test a.log_w == e.log_w
            @test a.ancestor === e.ancestor
        end
        @test actual[1].state.x == reference[1]
        @test old[1].state.x == original_x
        updated = GF._update_particles(model.obs, algo, 1, actual, y)
        expected_update = GF._population_map(execution, n) do i
            GF.update_particle(model.obs, algo, 1, actual[i], y)
        end
        for (a, e) in zip(updated, expected_update)
            @test a.state.x == e.state.x
            @test a.state.z.μ == e.state.z.μ
            @test a.state.z.Σ == e.state.z.Σ
            @test a.log_w == e.log_w
            @test a.ancestor === e.ancestor
        end
    end
end
