@testitem "Kalman raw core storage and cache contract" begin
    using GeneralisedFilters
    using LinearAlgebra
    using StaticArrays
    import GeneralisedFilters as GF

    # Mixed precision is the extraction-specific risk; same-precision numerical
    # and AD correctness are already covered by the existing Kalman suites.
    for static in (false, true)
        T = Float32
        v(x) = static ? SVector{length(x)}(x) : x
        m(x) = static ? SMatrix{size(x, 1),size(x, 2)}(x) : x
        # Mean and covariance deliberately differ in precision: normalisation
        # belongs at the existing CPU prediction boundary, before the update.
        state = GaussianState(v(T[0.2, -0.3]), Diagonal(v([1.2, 0.8])))
        d = LinearGaussianDynamics(
            m(T[0.9 0.1; 0 0.8]), v(T[0.1, 0]), Diagonal(v(T[0.2, 0.3]))
        )
        o = LinearGaussianObservation(
            m(reshape(T[1, 0.5], 1, 2)), v(T[0.2]), m(reshape(T[0.4], 1, 1))
        )
        y = v(T[0.7])
        raw = GF._kalman_predict_raw(state, d)
        pred = GF.kalman_predict(state, d)
        @test eltype(raw.μ) === T
        @test eltype(raw.Σ) === Float64
        @test eltype(pred.μ) === eltype(pred.Σ) === Float64
        @test pred.μ == raw.μ
        @test pred.Σ == raw.Σ
        @test pred.μ isa (static ? SVector : Vector)
        @test pred.Σ isa (static ? SMatrix : Matrix)

        # The cache retains the exact computational inputs/intermediates used
        # by the handwritten adjoint, rather than reconstructed equivalents.
        rawf, rawll, rawcache = GF._kalman_update_raw(pred, o, y)
        filt, ll, cache = GF.kalman_update_cached(pred, o, y)
        stepf, stepll, stepcache = GF.kalman_step_cached(state, d, o, y)
        @test filt.μ == rawf.μ == stepf.μ
        @test filt.Σ == rawf.Σ == stepf.Σ
        @test ll == rawll == stepll
        @test cache == rawcache
        @test cache.μ̂ === pred.μ
        @test cache.Σ̂ === pred.Σ
        @test cache.H === o.H
        @test stepcache.μ0 === state.μ
        @test stepcache.Σ0 === state.Σ
        @test stepcache.A === d.A
    end
end
