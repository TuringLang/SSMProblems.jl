@testitem "Kalman shared core storage and cache contract" begin
    using GeneralisedFilters
    using LinearAlgebra
    using StaticArrays
    import GeneralisedFilters as GF

    # Mixed precision exercises independent field types; same-precision numerical
    # and AD correctness are already covered by the existing Kalman suites.
    for static in (false, true)
        T = Float32
        v(x) = static ? SVector{length(x)}(x) : x
        m(x) = static ? SMatrix{size(x, 1),size(x, 2)}(x) : x
        # Mean and covariance deliberately differ in precision. Initialisation
        # selects full storage without promoting one to the other's scalar type.
        state = GaussianState(v(T[0.2, -0.3]), Diagonal(v([1.2, 0.8])))
        d = LinearGaussianDynamics(
            m(T[0.9 0.1; 0 0.8]), v(T[0.1, 0]), Diagonal(v(T[0.2, 0.3]))
        )
        o = LinearGaussianObservation(
            m(reshape(T[1, 0.5], 1, 2)), v(T[0.2]), m(reshape(T[0.4], 1, 1))
        )
        y = v(T[0.7])
        initial = GF._kalman_state(state.μ, state.Σ)
        @test eltype(initial.μ) === T
        @test eltype(initial.Σ) === Float64
        @test initial.Σ isa (static ? SMatrix : Matrix)
        pred = GF.kalman_predict(initial, d)
        @test eltype(pred.μ) === T
        @test eltype(pred.Σ) === Float64
        @test pred.μ isa (static ? SVector : Vector)
        @test pred.Σ isa (static ? SMatrix : Matrix)

        # The cache retains the exact computational inputs/intermediates used
        # by the handwritten adjoint, rather than reconstructed equivalents.
        filt, ll, cache = GF.kalman_update_cached(pred, o, y)
        stepf, stepll, stepcache = GF.kalman_step_cached(state, d, o, y)
        @test filt.μ == stepf.μ
        @test filt.Σ == stepf.Σ
        @test ll == stepll
        @test cache.μ̂ === pred.μ
        @test cache.Σ̂ === pred.Σ
        @test cache.H === o.H
        @test stepcache.μ0 === state.μ
        @test stepcache.Σ0 === state.Σ
        @test stepcache.A === d.A
    end
end
