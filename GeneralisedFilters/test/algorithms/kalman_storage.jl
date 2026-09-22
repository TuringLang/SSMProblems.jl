@testitem "Kalman structured state storage and history promotion" begin
    using GeneralisedFilters
    using LinearAlgebra
    using StaticArrays
    using Random

    ys = [[0.4, -0.7], [0.2, 0.8], [-0.1, 0.3]]
    A, H = [0.8 0.2; 0.0 0.8], Matrix{Float64}(I, 2, 2)
    for static in (false, true),
        covariance in (
            Diagonal([1.7, 1.2]),
            Symmetric([1.7 0.3; -0.6 1.2], :U),
            Symmetric([1.7 -0.6; 0.3 1.2], :L),
        )

        v(x) = static ? SVector{2}(x) : x
        m(x) = static ? SMatrix{2,2}(x) : x
        μ = v(zeros(2))
        model = create_homogeneous_linear_gaussian_model(
            μ,
            covariance,
            m(A),
            μ,
            Diagonal(v([0.5, 0.4])),
            m(H),
            μ,
            Diagonal(v([0.3, 0.2])),
        )
        dense = create_homogeneous_linear_gaussian_model(
            μ,
            m(Matrix(covariance)),
            m(A),
            μ,
            m(diagm([0.5, 0.4])),
            m(H),
            μ,
            m(diagm([0.3, 0.2])),
        )
        obs = v.(ys)
        original = copy(parent(covariance))
        initial = GeneralisedFilters.initialise(Random.default_rng(), model.prior, KF())
        @test initial.Σ == Matrix(covariance)
        @test initial.Σ isa (static ? SMatrix : Matrix)
        for t in 1:length(ys)
            got, ll = GeneralisedFilters.smooth(model, KS, obs; t_smooth=t)
            expected, expected_ll = GeneralisedFilters.smooth(dense, KS, obs; t_smooth=t)
            @test got ≈ expected
            @test ll ≈ expected_ll
            @test got.Σ isa (static ? SMatrix : Matrix)
        end
        @test marginal_loglikelihood(model, KF(), obs) ≈
            marginal_loglikelihood(dense, KF(), obs)
        @test model.prior.Σ0 === covariance
        @test parent(covariance) == original
    end

    # The first prediction stays Float32; the first update promotes to Float64.
    # Prior-based or first-prediction-based history types both fail this case.
    for static in (false, true)
        v(x) = static ? SVector{2}(x) : x
        m(x) = static ? SMatrix{2,2}(x) : x
        model = create_homogeneous_linear_gaussian_model(
            v(zeros(Float32, 2)),
            Diagonal(v(Float32[1, 1])),
            m(Float32.(A)),
            v(zeros(Float32, 2)),
            Diagonal(v(Float32[0.5, 0.4])),
            m(H),
            v(zeros(2)),
            Diagonal(v([0.3, 0.2])),
        )
        for T in (1, 3)
            obs = v.(ys[1:T])
            smoothed, ll = GeneralisedFilters.smooth(model, KS, obs)
            @test eltype(smoothed.μ) === Float64
            @test eltype(smoothed.Σ) === Float64
            @test ll ≈ marginal_loglikelihood(model, KF(), obs)
        end
        @test marginal_loglikelihood(model, KF(), Vector{typeof(v(ys[1]))}()) == 0
    end
end
