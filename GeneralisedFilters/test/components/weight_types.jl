@testitem "Explicit particle weight identities" begin
    using GeneralisedFilters
    using ForwardDiff
    using LogExpFunctions
    const GF = GeneralisedFilters

    z = GF.TypelessZero()
    @test !(z isa Number)
    @test !(GF.TypelessBaseline(3) isa Number)
    @test add_logweight(z, z) isa GF.TypelessZero
    @test_throws MethodError z + 1.0
    @test_throws MethodError convert(Float64, z)
    @test_throws MethodError add_logweight(z, "invalid density")

    for T in (Float16, Float32, Float64, BigFloat)
        x = T(0.125)
        @test add_logweight(z, x) === x
        @test add_logweight(x, z) === x
        @test add_logweight(x, x) == x + x
        @test GF._subtract_baseline(x, GF.TypelessBaseline(1)) == x
        @test GF._subtract_baseline(x, GF.TypelessBaseline(3)) isa T
        @test GF._add_baseline(GF.TypelessBaseline(3), x) isa T
    end
    @test add_logweight(1.0f0, 2.0) isa Float64
    @test GF._subtract_baseline(GF.TypelessBaseline(3), GF.TypelessBaseline(3)) isa
        GF.TypelessZero
    @test_throws ArgumentError GF._subtract_baseline(
        GF.TypelessBaseline(2), GF.TypelessBaseline(3)
    )
    @test isfinite(GF._log_count_like(Float16(0), 100_000))
    setprecision(BigFloat, 256) do
        expected = log(BigFloat(3))
        @test GF._log_count_like(BigFloat(0), 3) == expected
        @test abs(expected - BigFloat(log(3))) > big"1e-20"
    end

    objective(x) = GF._subtract_baseline(add_logweight(z, x^3), GF.TypelessBaseline(3))
    @test ForwardDiff.derivative(objective, 2.0) == 12.0
    @test ForwardDiff.derivative(x -> ForwardDiff.derivative(objective, x), 2.0) == 12.0
    @test ForwardDiff.derivative(objective, 2.0f0) isa Float32

    particles = [GF.Particle(i, 0) for i in 1:3]
    state = GF.ParticleDistribution(particles, GF.TypelessBaseline(3))
    @test GF.get_weights(state) == fill(1 / 3, 3)
    @test GF._weight_logsumexp(GF.log_weights(state)).N == 3
    for T in (Float32, Float64, BigFloat)
        weighted = [
            GF.Particle(p.state, T(-i), p.ancestor) for (i, p) in enumerate(particles)
        ]
        updated, ll = GF.marginalise!(state, weighted)
        @test ll isa T
        @test ll ≈ logsumexp(T.(-1:-1:-3)) - log(T(3))
        @test eltype(GF.log_weights(updated)) === T
        @test sum(GF.get_weights(updated)) ≈ one(T)
    end
end
