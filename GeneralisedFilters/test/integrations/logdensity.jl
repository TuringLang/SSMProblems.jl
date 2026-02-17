"""Tests for the log-density interface (trajectory_logdensity, kf_loglikelihood, rrule)."""

## Regular SSM trajectory_logdensity ###########################################################

@testitem "trajectory_logdensity: regular SSM" begin
    using GeneralisedFilters
    using SSMProblems
    using StableRNGs
    using Distributions
    using OffsetArrays

    let
        rng = StableRNG(1234)
        Dx, Dy, T = 2, 2, 5
        model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)

        x0, xs, ys = SSMProblems.sample(rng, model, T)
        trajectory = OffsetVector(vcat([x0], xs), -1)

        ll = trajectory_logdensity(model, trajectory, ys)

        ll_manual = logpdf(SSMProblems.distribution(SSMProblems.prior(model)), x0)
        for t in 1:T
            ll_manual += SSMProblems.logdensity(
                SSMProblems.dyn(model), t, trajectory[t - 1], trajectory[t]
            )
            ll_manual += SSMProblems.logdensity(
                SSMProblems.obs(model), t, trajectory[t], ys[t]
            )
        end

        @test ll ≈ ll_manual
    end
end

## HierarchicalSSM trajectory_logdensity #######################################################

@testitem "trajectory_logdensity: HierarchicalSSM" begin
    using GeneralisedFilters
    using SSMProblems
    using StableRNGs
    using Distributions
    using OffsetArrays

    let
        rng = StableRNG(1234)
        D_outer, D_inner, D_obs, T = 2, 2, 2, 5

        full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
            rng, D_outer, D_inner, D_obs; static_arrays=false
        )
        x0, z0, xs, zs, ys = SSMProblems.sample(rng, hier_model, T)
        outer_traj = OffsetVector(vcat([x0], xs), -1)

        ll = trajectory_logdensity(hier_model, KF(), outer_traj, ys)

        ll_manual = logpdf(SSMProblems.distribution(hier_model.outer_prior), outer_traj[0])
        for t in 1:T
            ll_manual += SSMProblems.logdensity(
                hier_model.outer_dyn, t, outer_traj[t - 1], outer_traj[t]
            )
        end

        inner_model = hier_model.inner_model
        state = GeneralisedFilters.initialise(
            rng, inner_model.prior, KF(); new_outer=outer_traj[0]
        )
        ll_inner = 0.0
        for t in 1:T
            state = GeneralisedFilters.predict(
                rng,
                inner_model.dyn,
                KF(),
                t,
                state,
                nothing;
                prev_outer=outer_traj[t - 1],
                new_outer=outer_traj[t],
            )
            state, ll_inc = GeneralisedFilters.update(
                inner_model.obs, KF(), t, state, ys[t]; new_outer=outer_traj[t]
            )
            ll_inner += ll_inc
        end
        ll_manual += ll_inner

        @test ll ≈ ll_manual
    end
end

## kf_loglikelihood value ######################################################################

@testitem "kf_loglikelihood value" begin
    using GeneralisedFilters
    using SSMProblems
    using StableRNGs
    using PDMats

    rng = StableRNG(1234)
    Dx, Dy, T = 2, 2, 5
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)
    _, _, ys = SSMProblems.sample(rng, model, T)

    # Extract parameters
    pr = SSMProblems.prior(model)
    dy = SSMProblems.dyn(model)
    ob = SSMProblems.obs(model)

    μ0 = GeneralisedFilters.calc_μ0(pr)
    Σ0 = GeneralisedFilters.calc_Σ0(pr)
    A = GeneralisedFilters.calc_A(dy, 1)
    b = GeneralisedFilters.calc_b(dy, 1)
    Q = GeneralisedFilters.calc_Q(dy, 1)
    H = GeneralisedFilters.calc_H(ob, 1)
    c = GeneralisedFilters.calc_c(ob, 1)
    R = GeneralisedFilters.calc_R(ob, 1)

    # Homogeneous: same params at each timestep
    As = fill(A, T)
    bs = fill(b, T)
    Qs = fill(Q, T)
    Hs = fill(H, T)
    cs = fill(c, T)
    Rs = fill(R, T)

    ll_kf = kf_loglikelihood(μ0, Σ0, As, bs, Qs, Hs, cs, Rs, ys)

    # Compare against filter()
    _, ll_filter = GeneralisedFilters.filter(model, KF(), ys)

    @test ll_kf ≈ ll_filter
end

## kf_loglikelihood rrule gradients ############################################################

@testitem "kf_loglikelihood rrule: ∂b" begin
    using GeneralisedFilters
    using SSMProblems
    using ChainRulesCore
    using FiniteDifferences
    using StableRNGs
    using PDMats

    rng = StableRNG(1234)
    Dx, Dy, T = 2, 2, 3
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)
    _, _, ys = SSMProblems.sample(rng, model, T)

    pr = SSMProblems.prior(model)
    dy = SSMProblems.dyn(model)
    ob = SSMProblems.obs(model)

    μ0 = Vector(GeneralisedFilters.calc_μ0(pr))
    Σ0 = PDMat(Matrix(GeneralisedFilters.calc_Σ0(pr)))
    A = Matrix(GeneralisedFilters.calc_A(dy, 1))
    b = Vector(GeneralisedFilters.calc_b(dy, 1))
    Q = PDMat(Matrix(GeneralisedFilters.calc_Q(dy, 1)))
    H = Matrix(GeneralisedFilters.calc_H(ob, 1))
    c = Vector(GeneralisedFilters.calc_c(ob, 1))
    R = PDMat(Matrix(GeneralisedFilters.calc_R(ob, 1)))
    ys_vec = [Vector(y) for y in ys]

    As = fill(A, T)
    Qs = fill(Q, T)
    Hs = fill(H, T)
    Rs = fill(R, T)
    cs_arr = fill(c, T)

    # rrule
    _, pullback = ChainRulesCore.rrule(
        kf_loglikelihood, μ0, Σ0, As, fill(b, T), Qs, Hs, cs_arr, Rs, ys_vec
    )
    cotangents = pullback(1.0)
    ∂bs_rrule = cotangents[5]  # index 5: bs (after NoTangent, μ0, Σ0, As)

    # Finite differences
    fdm = central_fdm(5, 1)
    function ll_b(b_vec)
        return kf_loglikelihood(μ0, Σ0, As, fill(b_vec, T), Qs, Hs, cs_arr, Rs, ys_vec)
    end
    ∂b_fd = FiniteDifferences.grad(fdm, ll_b, b)[1]

    # Sum rrule gradients over timesteps (homogeneous model → all the same)
    ∂b_rrule_total = sum(∂bs_rrule)
    @test ∂b_rrule_total ≈ ∂b_fd rtol = 1e-4
end

@testitem "kf_loglikelihood rrule: ∂c" begin
    using GeneralisedFilters
    using SSMProblems
    using ChainRulesCore
    using FiniteDifferences
    using StableRNGs
    using PDMats

    rng = StableRNG(1234)
    Dx, Dy, T = 2, 2, 3
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)
    _, _, ys = SSMProblems.sample(rng, model, T)

    pr = SSMProblems.prior(model)
    dy = SSMProblems.dyn(model)
    ob = SSMProblems.obs(model)

    μ0 = Vector(GeneralisedFilters.calc_μ0(pr))
    Σ0 = PDMat(Matrix(GeneralisedFilters.calc_Σ0(pr)))
    A = Matrix(GeneralisedFilters.calc_A(dy, 1))
    b = Vector(GeneralisedFilters.calc_b(dy, 1))
    Q = PDMat(Matrix(GeneralisedFilters.calc_Q(dy, 1)))
    H = Matrix(GeneralisedFilters.calc_H(ob, 1))
    c = Vector(GeneralisedFilters.calc_c(ob, 1))
    R = PDMat(Matrix(GeneralisedFilters.calc_R(ob, 1)))
    ys_vec = [Vector(y) for y in ys]

    As = fill(A, T)
    bs = fill(b, T)
    Qs = fill(Q, T)
    Hs = fill(H, T)
    Rs = fill(R, T)

    _, pullback = ChainRulesCore.rrule(
        kf_loglikelihood, μ0, Σ0, As, bs, Qs, Hs, fill(c, T), Rs, ys_vec
    )
    cotangents = pullback(1.0)
    ∂cs_rrule = cotangents[8]  # index 8: cs

    fdm = central_fdm(5, 1)
    function ll_c(c_vec)
        return kf_loglikelihood(μ0, Σ0, As, bs, Qs, Hs, fill(c_vec, T), Rs, ys_vec)
    end
    ∂c_fd = FiniteDifferences.grad(fdm, ll_c, c)[1]

    ∂c_rrule_total = sum(∂cs_rrule)
    @test ∂c_rrule_total ≈ ∂c_fd rtol = 1e-4
end

@testitem "kf_loglikelihood rrule: ∂A" begin
    using GeneralisedFilters
    using SSMProblems
    using ChainRulesCore
    using FiniteDifferences
    using StableRNGs
    using PDMats

    rng = StableRNG(1234)
    Dx, Dy, T = 2, 2, 3
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)
    _, _, ys = SSMProblems.sample(rng, model, T)

    pr = SSMProblems.prior(model)
    dy = SSMProblems.dyn(model)
    ob = SSMProblems.obs(model)

    μ0 = Vector(GeneralisedFilters.calc_μ0(pr))
    Σ0 = PDMat(Matrix(GeneralisedFilters.calc_Σ0(pr)))
    A = Matrix(GeneralisedFilters.calc_A(dy, 1))
    b = Vector(GeneralisedFilters.calc_b(dy, 1))
    Q = PDMat(Matrix(GeneralisedFilters.calc_Q(dy, 1)))
    H = Matrix(GeneralisedFilters.calc_H(ob, 1))
    c = Vector(GeneralisedFilters.calc_c(ob, 1))
    R = PDMat(Matrix(GeneralisedFilters.calc_R(ob, 1)))
    ys_vec = [Vector(y) for y in ys]

    bs = fill(b, T)
    Qs = fill(Q, T)
    Hs = fill(H, T)
    cs = fill(c, T)
    Rs = fill(R, T)

    _, pullback = ChainRulesCore.rrule(
        kf_loglikelihood, μ0, Σ0, fill(A, T), bs, Qs, Hs, cs, Rs, ys_vec
    )
    cotangents = pullback(1.0)
    ∂As_rrule = cotangents[4]

    fdm = central_fdm(5, 1)
    function ll_A(A_vec)
        A_mat = reshape(A_vec, size(A))
        return kf_loglikelihood(μ0, Σ0, fill(A_mat, T), bs, Qs, Hs, cs, Rs, ys_vec)
    end
    ∂A_fd = reshape(FiniteDifferences.grad(fdm, ll_A, vec(A))[1], size(A))

    ∂A_rrule_total = sum(∂As_rrule)
    @test ∂A_rrule_total ≈ ∂A_fd rtol = 1e-4
end

@testitem "kf_loglikelihood rrule: ∂μ0" begin
    using GeneralisedFilters
    using SSMProblems
    using ChainRulesCore
    using FiniteDifferences
    using StableRNGs
    using PDMats

    rng = StableRNG(1234)
    Dx, Dy, T = 2, 2, 3
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)
    _, _, ys = SSMProblems.sample(rng, model, T)

    pr = SSMProblems.prior(model)
    dy = SSMProblems.dyn(model)
    ob = SSMProblems.obs(model)

    μ0 = Vector(GeneralisedFilters.calc_μ0(pr))
    Σ0 = PDMat(Matrix(GeneralisedFilters.calc_Σ0(pr)))
    A = Matrix(GeneralisedFilters.calc_A(dy, 1))
    b = Vector(GeneralisedFilters.calc_b(dy, 1))
    Q = PDMat(Matrix(GeneralisedFilters.calc_Q(dy, 1)))
    H = Matrix(GeneralisedFilters.calc_H(ob, 1))
    c = Vector(GeneralisedFilters.calc_c(ob, 1))
    R = PDMat(Matrix(GeneralisedFilters.calc_R(ob, 1)))
    ys_vec = [Vector(y) for y in ys]

    As = fill(A, T)
    bs = fill(b, T)
    Qs = fill(Q, T)
    Hs = fill(H, T)
    cs = fill(c, T)
    Rs = fill(R, T)

    _, pullback = ChainRulesCore.rrule(
        kf_loglikelihood, μ0, Σ0, As, bs, Qs, Hs, cs, Rs, ys_vec
    )
    cotangents = pullback(1.0)
    ∂μ0_rrule = cotangents[2]

    fdm = central_fdm(5, 1)
    function ll_μ0(μ0_vec)
        return kf_loglikelihood(μ0_vec, Σ0, As, bs, Qs, Hs, cs, Rs, ys_vec)
    end
    ∂μ0_fd = FiniteDifferences.grad(fdm, ll_μ0, μ0)[1]

    @test ∂μ0_rrule ≈ ∂μ0_fd rtol = 1e-4
end

@testitem "kf_loglikelihood rrule: ∂Q" begin
    using GeneralisedFilters
    using SSMProblems
    using ChainRulesCore
    using FiniteDifferences
    using StableRNGs
    using PDMats
    using LinearAlgebra

    rng = StableRNG(1234)
    Dx, Dy, T = 2, 2, 3
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)
    _, _, ys = SSMProblems.sample(rng, model, T)

    pr = SSMProblems.prior(model)
    dy = SSMProblems.dyn(model)
    ob = SSMProblems.obs(model)

    μ0 = Vector(GeneralisedFilters.calc_μ0(pr))
    Σ0 = PDMat(Matrix(GeneralisedFilters.calc_Σ0(pr)))
    A = Matrix(GeneralisedFilters.calc_A(dy, 1))
    b = Vector(GeneralisedFilters.calc_b(dy, 1))
    Q = PDMat(Matrix(GeneralisedFilters.calc_Q(dy, 1)))
    H = Matrix(GeneralisedFilters.calc_H(ob, 1))
    c = Vector(GeneralisedFilters.calc_c(ob, 1))
    R = PDMat(Matrix(GeneralisedFilters.calc_R(ob, 1)))
    ys_vec = [Vector(y) for y in ys]

    As = fill(A, T)
    bs = fill(b, T)
    Hs = fill(H, T)
    cs = fill(c, T)
    Rs = fill(R, T)

    _, pullback = ChainRulesCore.rrule(
        kf_loglikelihood, μ0, Σ0, As, bs, fill(Q, T), Hs, cs, Rs, ys_vec
    )
    cotangents = pullback(1.0)
    ∂Qs_rrule = cotangents[6]

    function make_pd(M, D)
        M_sym = (M + M') / 2
        λ, V = eigen(M_sym)
        λ_clipped = max.(λ, 1e-8)
        return PDMat(Symmetric(V * Diagonal(λ_clipped) * V'))
    end

    fdm = central_fdm(5, 1)
    function ll_Q(Q_vec)
        Q_new = make_pd(reshape(Q_vec, Dx, Dx), Dx)
        return kf_loglikelihood(μ0, Σ0, As, bs, fill(Q_new, T), Hs, cs, Rs, ys_vec)
    end
    ∂Q_fd = reshape(FiniteDifferences.grad(fdm, ll_Q, vec(Matrix(Q)))[1], Dx, Dx)

    ∂Q_rrule_total = sum(∂Qs_rrule)
    @test ∂Q_rrule_total ≈ ∂Q_fd rtol = 1e-3
end

@testitem "kf_loglikelihood rrule: ∂R" begin
    using GeneralisedFilters
    using SSMProblems
    using ChainRulesCore
    using FiniteDifferences
    using StableRNGs
    using PDMats
    using LinearAlgebra

    rng = StableRNG(1234)
    Dx, Dy, T = 2, 2, 3
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)
    _, _, ys = SSMProblems.sample(rng, model, T)

    pr = SSMProblems.prior(model)
    dy = SSMProblems.dyn(model)
    ob = SSMProblems.obs(model)

    μ0 = Vector(GeneralisedFilters.calc_μ0(pr))
    Σ0 = PDMat(Matrix(GeneralisedFilters.calc_Σ0(pr)))
    A = Matrix(GeneralisedFilters.calc_A(dy, 1))
    b = Vector(GeneralisedFilters.calc_b(dy, 1))
    Q = PDMat(Matrix(GeneralisedFilters.calc_Q(dy, 1)))
    H = Matrix(GeneralisedFilters.calc_H(ob, 1))
    c = Vector(GeneralisedFilters.calc_c(ob, 1))
    R = PDMat(Matrix(GeneralisedFilters.calc_R(ob, 1)))
    ys_vec = [Vector(y) for y in ys]

    As = fill(A, T)
    bs = fill(b, T)
    Qs = fill(Q, T)
    Hs = fill(H, T)
    cs = fill(c, T)

    _, pullback = ChainRulesCore.rrule(
        kf_loglikelihood, μ0, Σ0, As, bs, Qs, Hs, cs, fill(R, T), ys_vec
    )
    cotangents = pullback(1.0)
    ∂Rs_rrule = cotangents[9]

    function make_pd(M, D)
        M_sym = (M + M') / 2
        λ, V = eigen(M_sym)
        λ_clipped = max.(λ, 1e-8)
        return PDMat(Symmetric(V * Diagonal(λ_clipped) * V'))
    end

    fdm = central_fdm(5, 1)
    function ll_R(R_vec)
        R_new = make_pd(reshape(R_vec, Dy, Dy), Dy)
        return kf_loglikelihood(μ0, Σ0, As, bs, Qs, Hs, cs, fill(R_new, T), ys_vec)
    end
    ∂R_fd = reshape(FiniteDifferences.grad(fdm, ll_R, vec(Matrix(R)))[1], Dy, Dy)

    ∂R_rrule_total = sum(∂Rs_rrule)
    @test ∂R_rrule_total ≈ ∂R_fd rtol = 1e-4
end

@testitem "kf_loglikelihood rrule: ∂H" begin
    using GeneralisedFilters
    using SSMProblems
    using ChainRulesCore
    using FiniteDifferences
    using StableRNGs
    using PDMats

    rng = StableRNG(1234)
    Dx, Dy, T = 2, 2, 3
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)
    _, _, ys = SSMProblems.sample(rng, model, T)

    pr = SSMProblems.prior(model)
    dy = SSMProblems.dyn(model)
    ob = SSMProblems.obs(model)

    μ0 = Vector(GeneralisedFilters.calc_μ0(pr))
    Σ0 = PDMat(Matrix(GeneralisedFilters.calc_Σ0(pr)))
    A = Matrix(GeneralisedFilters.calc_A(dy, 1))
    b = Vector(GeneralisedFilters.calc_b(dy, 1))
    Q = PDMat(Matrix(GeneralisedFilters.calc_Q(dy, 1)))
    H = Matrix(GeneralisedFilters.calc_H(ob, 1))
    c = Vector(GeneralisedFilters.calc_c(ob, 1))
    R = PDMat(Matrix(GeneralisedFilters.calc_R(ob, 1)))
    ys_vec = [Vector(y) for y in ys]

    As = fill(A, T)
    bs = fill(b, T)
    Qs = fill(Q, T)
    cs = fill(c, T)
    Rs = fill(R, T)

    _, pullback = ChainRulesCore.rrule(
        kf_loglikelihood, μ0, Σ0, As, bs, Qs, fill(H, T), cs, Rs, ys_vec
    )
    cotangents = pullback(1.0)
    ∂Hs_rrule = cotangents[7]

    fdm = central_fdm(5, 1)
    function ll_H(H_vec)
        H_mat = reshape(H_vec, size(H))
        return kf_loglikelihood(μ0, Σ0, As, bs, Qs, fill(H_mat, T), cs, Rs, ys_vec)
    end
    ∂H_fd = reshape(FiniteDifferences.grad(fdm, ll_H, vec(H))[1], size(H))

    ∂H_rrule_total = sum(∂Hs_rrule)
    @test ∂H_rrule_total ≈ ∂H_fd rtol = 1e-4
end

## kf_loglikelihood with StaticArrays ##########################################################

@testitem "kf_loglikelihood: StaticArrays value + rrule" begin
    using GeneralisedFilters
    using SSMProblems
    using ChainRulesCore
    using FiniteDifferences
    using StableRNGs
    using PDMats
    using StaticArrays
    using LinearAlgebra

    rng = StableRNG(1234)
    Dx, Dy, T = 2, 2, 3

    # Build static array parameters
    μ0 = @SVector randn(rng, Dx)
    Σ0_mat = let M = @SMatrix randn(rng, Dx, Dx)
        PDMat(Symmetric(M * M' + 0.1I))
    end
    A = @SMatrix randn(rng, Dx, Dx)
    b = @SVector randn(rng, Dx)
    Q = let M = @SMatrix randn(rng, Dx, Dx)
        PDMat(Symmetric(M * M' + 0.1I))
    end
    H = @SMatrix randn(rng, Dy, Dx)
    c = @SVector randn(rng, Dy)
    R = let M = @SMatrix randn(rng, Dy, Dy)
        PDMat(Symmetric(M * M' + 0.1I))
    end

    As = fill(A, T)
    bs_arr = fill(b, T)
    Qs = fill(Q, T)
    Hs = fill(H, T)
    cs_arr = fill(c, T)
    Rs = fill(R, T)

    # Generate observations
    ys = [SVector{Dy}(randn(rng, Dy)) for _ in 1:T]

    # Value: compare against regular arrays
    ll_static = kf_loglikelihood(μ0, Σ0_mat, As, bs_arr, Qs, Hs, cs_arr, Rs, ys)
    ll_dense = kf_loglikelihood(
        Vector(μ0),
        PDMat(Matrix(Σ0_mat)),
        [Matrix(A) for A in As],
        [Vector(b) for b in bs_arr],
        [PDMat(Matrix(Q)) for Q in Qs],
        [Matrix(H) for H in Hs],
        [Vector(c) for c in cs_arr],
        [PDMat(Matrix(R)) for R in Rs],
        [Vector(y) for y in ys],
    )
    @test ll_static ≈ ll_dense

    # rrule: check output types are static
    _, pullback = ChainRulesCore.rrule(
        kf_loglikelihood, μ0, Σ0_mat, As, bs_arr, Qs, Hs, cs_arr, Rs, ys
    )
    cotangents = pullback(1.0)
    ∂μ0 = cotangents[2]
    ∂Σ0 = cotangents[3]
    ∂As = cotangents[4]
    ∂bs = cotangents[5]

    @test ∂μ0 isa SVector{Dx}
    @test ∂Σ0 isa SMatrix{Dx,Dx}
    @test eltype(∂As) <: SMatrix{Dx,Dx}
    @test eltype(∂bs) <: SVector{Dx}

    # rrule: check gradient correctness for b
    fdm = central_fdm(5, 1)
    function ll_b(b_vec)
        b_s = SVector{Dx}(b_vec)
        return kf_loglikelihood(μ0, Σ0_mat, As, fill(b_s, T), Qs, Hs, cs_arr, Rs, ys)
    end
    ∂b_fd = FiniteDifferences.grad(fdm, ll_b, Vector(b))[1]
    @test sum(∂bs) ≈ SVector{Dx}(∂b_fd) rtol = 1e-4
end

## SSMParameterLogDensity ######################################################################

@testitem "SSMParameterLogDensity: regular SSM" begin
    using GeneralisedFilters
    using SSMProblems
    using LogDensityProblems
    using StableRNGs
    using Distributions
    using PDMats
    using LinearAlgebra
    using OffsetArrays

    rng = StableRNG(1234)

    # Simple 1D model: x_t = a * x_{t-1} + b + noise, y_t = x_t + noise
    # Unknown parameter: b (drift)
    a = 0.8
    q² = 0.1
    r² = 0.5
    σ₀² = 1.0
    T_len = 5

    function build_ssm(θ)
        return create_homogeneous_linear_gaussian_model(
            [0.0],
            PDMat([σ₀²;;]),
            [a;;],
            [θ[1]],
            PDMat([q²;;]),
            [1.0;;],
            [0.0],
            PDMat([r²;;]),
        )
    end

    true_b = 1.0
    true_ssm = build_ssm([true_b])
    _, _, ys = SSMProblems.sample(rng, true_ssm, T_len)

    # Sample a trajectory
    x0, xs, _ = SSMProblems.sample(rng, true_ssm, T_len)
    trajectory = OffsetVector(vcat([x0], xs), -1)

    prior = MvNormal([0.0], [4.0;;])
    pssm = ParameterisedSSM(build_ssm, ys)
    ld = SSMParameterLogDensity(prior, pssm, Ref(trajectory))

    θ_test = [0.5]
    ll = LogDensityProblems.logdensity(ld, θ_test)

    # Manual
    model = build_ssm(θ_test)
    ll_expected = logpdf(prior, θ_test) + trajectory_logdensity(model, trajectory, ys)

    @test ll ≈ ll_expected
    @test LogDensityProblems.dimension(ld) == 1
end

@testitem "SSMParameterLogDensity: HierarchicalSSM" begin
    using GeneralisedFilters
    using SSMProblems
    using LogDensityProblems
    using StableRNGs
    using Distributions
    using PDMats
    using LinearAlgebra
    using OffsetArrays

    rng = StableRNG(1234)

    D_outer, D_inner, D_obs = 1, 1, 1
    T_len = 5

    _, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
        rng, D_outer, D_inner, D_obs; static_arrays=false
    )
    x0, _, xs, _, ys = SSMProblems.sample(rng, hier_model, T_len)
    outer_traj = OffsetVector(vcat([x0], xs), -1)

    # Parameterise the model by b (inner dynamics offset)
    true_b = hier_model.inner_model.dyn.b
    fixed_model = hier_model

    function build_hier(θ)
        inner_dyn = GeneralisedFilters.GFTest.InnerDynamics(
            fixed_model.inner_model.dyn.A,
            θ,
            fixed_model.inner_model.dyn.C,
            fixed_model.inner_model.dyn.Q,
        )
        return HierarchicalSSM(
            fixed_model.outer_prior,
            fixed_model.outer_dyn,
            fixed_model.inner_model.prior,
            inner_dyn,
            fixed_model.inner_model.obs,
        )
    end

    prior = MvNormal(zeros(D_inner), 4.0 * I)
    pssm = ParameterisedSSM(build_hier, ys)
    ld = SSMParameterLogDensity(prior, pssm, KF(), Ref(outer_traj))

    θ_test = [0.5]
    ll = LogDensityProblems.logdensity(ld, θ_test)

    model = build_hier(θ_test)
    ll_expected = logpdf(prior, θ_test) + trajectory_logdensity(model, KF(), outer_traj, ys)

    @test ll ≈ ll_expected
    @test LogDensityProblems.dimension(ld) == 1
end
