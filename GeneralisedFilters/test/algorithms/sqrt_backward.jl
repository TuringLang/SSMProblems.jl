@testitem "Square-root backward likelihood and severe precision regression" begin
    using GeneralisedFilters, StaticArrays, LinearAlgebra
    G = GeneralisedFilters
    d = LinearGaussianDynamics(SMatrix{1,1}(1.0), SVector(0.0), SMatrix{1,1}(1.0))
    legacy = InformationLikelihood(SVector(0.0), SMatrix{1,1}(1e20))
    @test G.natural_params(
        G.backward_predict(BackwardInformationPredictor(), legacy, d)
    )[2][1] ≈ 1.0
    bp = SqrtBackwardInformationPredictor()
    o = LinearGaussianObservation(SMatrix{1,1}(1.0), SVector(0.0), SMatrix{1,1}(1e-20))
    l = G.backward_predict(bp, G.backward_initialise(bp, o, SVector(0.0)), d)
    @test G.natural_params(l)[2][1] ≈ 1.0
    @test l.B isa SMatrix
    @test G.compute_marginal_predictive_likelihood(
        GaussianState(SVector(0.0), SMatrix{1,1}(1.0)), l; include_constant=true
    ) ≈ -log(4π) / 2

    # Full normalized suffix likelihood, including QR-discarded residual constants.
    for static in (false, true)
        mat(A) = static ? SMatrix{size(A, 1),size(A, 2)}(A) : Matrix(A)
        vec(a) = static ? SVector{length(a)}(a) : Vector(a)
        p = GaussianPrior(vec([0.3, -0.2]), mat([1.0 0.2; 0.2 0.8]))
        dyn = LinearGaussianDynamics(
            mat([0.8 0.2; -0.1 0.7]), vec([0.1, -0.1]), CovarianceFactor(mat([0.3; 0.1;;]))
        )
        obs = LinearGaussianObservation(
            mat([1.0 0.1; 0.0 0.8; -0.2 0.3]),
            vec([0.2, -0.1, 0.05]),
            mat(Matrix(0.4I, 3, 3)),
        )
        ys = [vec([0.1t, -0.2t, 0.15]) for t in 1:7]
        model = StateSpaceModel(p, dyn, obs)
        back = G.backward_initialise(bp, obs, ys[end])
        for t in (length(ys) - 1):-1:1
            back = G.backward_update(bp, G.backward_predict(bp, back, dyn), obs, ys[t])
        end
        prior = G.initialise(G.default_rng(), p, SRKF())
        pred = G.srkf_predict(prior, dyn)
        @test G.compute_marginal_predictive_likelihood(pred, back; include_constant=true) ≈
            marginal_loglikelihood(model, SRKF(), ys) rtol = 1e-11
        shifted = SqrtInformationLikelihood(back.B, back.r, -1e100)
        @test G.compute_marginal_predictive_likelihood(pred, back) ==
            G.compute_marginal_predictive_likelihood(pred, shifted)
        @test size(back.B) == (2, 2)
        @test !static || back.B isa SMatrix
    end
end

@testitem "Explicit singular covariance factors preserve the model" begin
    using GeneralisedFilters, StaticArrays, LinearAlgebra, Statistics
    G = GeneralisedFilters
    F0 = @SMatrix zeros(2, 1)
    F = SMatrix{2,1}(0.2, 0.0)
    p = GaussianPrior(SVector(0.0, 0.0), CovarianceFactor(F0))
    d = LinearGaussianDynamics(
        @SMatrix([0.8 0.2; 0.1 0.9]), SVector(0.0, 0.0), CovarianceFactor(F)
    )
    o = LinearGaussianObservation(@SMatrix([1.0 0.2]), SVector(0.0), SMatrix{1,1}(0.1))
    ys = [SVector(sin(t)) for t in 1:20]
    model = StateSpaceModel(p, d, o)
    covariance_model = StateSpaceModel(
        GaussianPrior(p.μ0, F0 * F0'), LinearGaussianDynamics(d.A, d.b, F * F'), o
    )
    state, ll = G.filter(model, SRKF(), ys)
    @test ll ≈ marginal_loglikelihood(covariance_model, KF(), ys) rtol = 1e-11
    @test rand(G.distribution(p)) isa SVector{2}
    @test state.U isa UpperTriangular{Float64,<:SMatrix}
    @test eigmin(Symmetric(cov(state))) >= 0
    # A zero diagonal must not zero an entire nonzero row during QR sign normalization.
    R = [0.0 1.0; 0.0 0.0]
    @test G._correct_cholesky_sign(R)' * G._correct_cholesky_sign(R) == R' * R
end

@testitem "Square-root conditional gradients across all Gaussian fields" tags = [:mooncake] begin
    using GeneralisedFilters, StaticArrays, LinearAlgebra, ForwardDiff, Mooncake
    using GeneralisedFilters.GFTest: check_gradients
    for static in (false, true)
        mat(A) = static ? SMatrix{size(A, 1),size(A, 2)}(A) : Matrix(A)
        vec(a) = static ? SVector{length(a)}(a) : Vector(a)
        op = GaussianPrior(SVector(0.0), SMatrix{1,1}(1.0))
        od = LinearGaussianDynamics(SMatrix{1,1}(0.9), SVector(0.0), SMatrix{1,1}(0.2))
        path = [SVector(0.1t) for t in 0:3]
        ys = [vec([0.1, -0.3]), vec([-0.2, 0.1]), vec([0.3, 0.4])]
        function objective(θ, af)
            ip = GaussianPrior(
                vec([θ[1], 0.0]), CovarianceFactor(mat([exp(θ[2]) 0.0; 0.1 0.8]))
            )
            id =
                ctx -> LinearGaussianDynamics(
                    mat([θ[3] 0.1; 0.0 0.8]),
                    vec([θ[4] * ctx.x_new[1], 0.1]),
                    CovarianceFactor(mat(reshape([exp(θ[5]), 0.05], 2, 1))),
                )
            io =
                ctx -> LinearGaussianObservation(
                    mat([θ[6] 0.0; 0.1 1.0]),
                    vec([θ[7], ctx.x[1]]),
                    CovarianceFactor(mat([exp(θ[8]) 0.0; 0.02 0.5])),
                )
            return trajectory_logdensity(StateSpaceModel(op, od, ip, id, io), af, path, ys)
        end
        θ = [0.1, -0.2, 0.7, 0.2, -1.0, 1.2, 0.1, -0.5]
        f = θ -> objective(θ, SRKF())
        @test isfinite(f(θ))
        @test check_gradients(f, θ; rtol=2e-5).agrees
        @test ForwardDiff.gradient(f, θ) ≈ GeneralisedFilters.GFTest.central_diff(f, θ) rtol =
            2e-5
    end
end

@testitem "Dense square-root QR reverse rule" tags = [:mooncake] begin
    using GeneralisedFilters, Mooncake, LinearAlgebra, ForwardDiff
    using GeneralisedFilters.GFTest: check_gradients, central_diff
    M = [1.0 0.2 -0.3; 0.1 1.4 0.5; -0.2 0.3 1.2; 0.7 -0.4 0.1; 0.2 0.5 -0.6]
    seed = [0.3 -0.2 0.8; 0.4 0.7 -0.6; 0.1 0.2 -0.9]
    f(x) = sum(seed .* GeneralisedFilters._qr_upper(reshape(x, 5, 3)))
    @test check_gradients(f, vec(M); rtol=1e-6).agrees
    @test ForwardDiff.gradient(f, vec(M)) ≈ central_diff(f, vec(M)) rtol = 1e-6
end

@testitem "Ill-conditioned multivariate backward weights versus high precision replay" begin
    using GeneralisedFilters, LinearAlgebra
    G = GeneralisedFilters
    setprecision(BigFloat, 256) do
        A = [0.9 0.2; -0.1 0.8]
        F = [1.0 1e-6; 0.7 -1e-6]
        H = [1.0 0.3; -0.2 0.8]
        d = LinearGaussianDynamics(A, zeros(2), CovarianceFactor(F))
        o = LinearGaussianObservation(H, zeros(2), 1e-16 * Matrix{Float64}(I, 2, 2))
        z = [0.2, -0.1]
        ys = [H * (A^t) * z for t in 1:4]
        bp = SqrtBackwardInformationPredictor()
        back = G.backward_initialise(bp, o, ys[end])
        for t in 3:-1:1
            back = G.backward_update(bp, G.backward_predict(bp, back, d), o, ys[t])
        end
        means = [[0.0, 0.0], [0.3, -0.1], [-0.2, 0.4]]
        P = [0.8 0.1; 0.1 0.5]
        db = LinearGaussianDynamics(
            BigFloat.(A), zeros(BigFloat, 2), BigFloat.(F) * BigFloat.(F)'
        )
        ob = LinearGaussianObservation(BigFloat.(H), zeros(BigFloat, 2), BigFloat.(o.R))
        replay = [
            marginal_loglikelihood(
                StateSpaceModel(GaussianPrior(BigFloat.(μ), BigFloat.(P)), db, ob),
                KF(),
                map(y -> BigFloat.(y), ys),
            ) for μ in means
        ]
        weights = [
            G.compute_marginal_predictive_likelihood(
                G.srkf_predict(
                    G.initialise(G.default_rng(), GaussianPrior(μ, P), SRKF()), d
                ),
                back,
            ) for μ in means
        ]
        @test weights .- weights[1] ≈ Float64.(replay .- replay[1]) rtol = 2e-6 atol = 2e-7
    end
end

@testitem "Static square-root updates infer dimensions and mixed scalar types" begin
    using GeneralisedFilters, StaticArrays, ForwardDiff, LinearAlgebra
    G = GeneralisedFilters
    p = GaussianPrior(SA[0.0, 0.0], SA[1.0 0.0; 0.0 1.0])
    state = G.initialise(G.default_rng(), p, SRKF())
    o = LinearGaussianObservation(SMatrix{1,2}(1.0, 0.5), SA[0.0], SMatrix{1,1}(0.3))
    y = SA[0.1]
    @test (@inferred G.srkf_update(state, o, y))[1].U isa UpperTriangular{Float64,<:SMatrix}
    allocation_probe(state, o, y) = @allocated GeneralisedFilters.srkf_update(state, o, y)
    allocation_probe(state, o, y)
    @test allocation_probe(state, o, y) <= 128
    # The prior/noise factors remain Float64 while H, or A, contains Duals.
    h = ForwardDiff.Dual(1.0, 1.0)
    active_o = LinearGaussianObservation(SMatrix{1,2}(h, 0.5), SA[0.0], o.R)
    dual_state, dual_ll = @inferred G.srkf_update(state, active_o, y)
    @test dual_state.U isa UpperTriangular{<:ForwardDiff.Dual,<:SMatrix}
    @test isfinite(ForwardDiff.partials(dual_ll)[1])
    d = LinearGaussianDynamics(
        SMatrix{2,2}(h, 0.0, 0.1, 0.9), SA[0.0, 0.0], SA[0.1 0.0; 0.0 0.2]
    )
    predicted = @inferred G.srkf_predict(state, d)
    @test predicted.U isa UpperTriangular{<:ForwardDiff.Dual,<:SMatrix}
end
