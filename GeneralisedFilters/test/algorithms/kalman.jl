"""Unit tests for Kalman filter and smoother algorithms."""

## Forward Filtering ########################################################################

@testitem "Kalman filter" begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs

    SEED = 1234
    Dx = 3
    Dys = [2, 3, 4]

    for Dy in Dys
        rng = StableRNG(1234)
        model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)
        _, _, ys = simulate(rng, model, 1)

        filtered, ll = GeneralisedFilters.filter(rng, model, KalmanFilter(), ys)

        # Let Z = [X0, X1, Y1] be the joint state vector
        μ_Z, Σ_Z = GeneralisedFilters.GFTest._compute_joint(model, 1)

        # Condition on observations using formula for MVN conditional distribution. See:
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        y = only(ys)
        I_x = (Dx + 1):(2Dx)
        I_y = (2Dx + 1):(2Dx + Dy)
        μ_X1 = μ_Z[I_x] + Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ (y - μ_Z[I_y]))
        Σ_X1 = Σ_Z[I_x, I_x] - Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ Σ_Z[I_y, I_x])

        @test filtered.μ ≈ μ_X1
        @test filtered.Σ ≈ Σ_X1

        # Exact marginal distribution to test log-likelihood
        μ_Y1 = μ_Z[I_y]
        Σ_Y1 = Σ_Z[I_y, I_y]
        LinearAlgebra.hermitianpart!(Σ_Y1)
        true_ll = logpdf(MvNormal(μ_Y1, Σ_Y1), y)
        @test ll ≈ true_ll
    end
end

@testitem "Kalman filter StaticArrays" begin
    using GeneralisedFilters
    using StableRNGs
    using StaticArrays

    D = 2
    rng = StableRNG(1234)

    μ0 = @SVector rand(rng, D)
    Σ0 = @SMatrix rand(rng, D, D)
    Σ0 = Σ0 * Σ0'
    A = @SMatrix rand(rng, D, D)
    b = @SVector rand(rng, D)
    Q = @SMatrix rand(rng, D, D)
    Q = Q * Q'
    H = @SMatrix rand(rng, D, D)
    c = @SVector rand(rng, D)
    R = @SMatrix rand(rng, D, D)
    R = R * R'

    model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)

    _, _, ys = simulate(rng, model, 2)

    state, _ = GeneralisedFilters.filter(rng, model, KalmanFilter(), ys)

    # Verify returned values are still StaticArrays
    @test ys[2] isa SVector{D,Float64}
    @test state.μ isa SVector{D,Float64}
    @test state.Σ isa SMatrix{D,D,Float64}
end

@testitem "Marginal log-likelihood" begin
    using GeneralisedFilters
    using StableRNGs
    using StaticArrays

    rng = StableRNG(1234)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(
        rng, 3, 2; static_arrays=true
    )
    _, _, ys = simulate(rng, model, 5)

    _, ll = GeneralisedFilters.filter(rng, model, KalmanFilter(), ys)
    mll = GeneralisedFilters.marginal_loglikelihood(model, KalmanFilter(), ys)

    @test mll ≈ ll
end

## Backward Information Filtering ###########################################################

@testitem "Backward information predictor" begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs

    SEED = 1234
    Dx = 3
    Dys = [2, 3, 4]
    T = 4

    for Dy in Dys
        rng = StableRNG(1234)
        model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)
        _, _, ys = simulate(rng, model, T)

        # Model is homogeneous, so the resolved atom is the same at every step; resolving at
        # the connecting index keeps the pattern valid for time-varying models too.
        res = GeneralisedFilters.resolve
        BIF = BackwardInformationPredictor(; initial_jitter=1e-8)
        pl = backward_initialise(BIF, res(model.obs, (; t=T)), ys[T])
        pl = backward_predict(BIF, pl, res(model.dyn, (; t=T)))
        pl = backward_update(BIF, pl, res(model.obs, (; t=T - 1)), ys[T - 1])
        pl = backward_predict(BIF, pl, res(model.dyn, (; t=T - 1)))
        pl = backward_update(BIF, pl, res(model.obs, (; t=T - 2)), ys[T - 2])

        # Assuming homogeneous
        A, b, Q = model.dyn.A, model.dyn.b, model.dyn.Q
        H, c, R = model.obs.H, model.obs.c, model.obs.R
        F = [H; H * A; H * A^2]
        g = [c; H * b + c; H * (A * b + b) + c]

        #! format: off
        Σ = [
            R               zeros(Dy, Dy)    zeros(Dy, Dy);
            zeros(Dy, Dy)   H * Q * H' + R   H * Q * A' * H';
            zeros(Dy, Dy)   H * A * Q * H'   H * (A * Q * A' + Q) * H' + R
        ]
        #! format: on

        λ_true = F' * inv(Σ) * (vcat(ys[(T - 2):T]...) .- g)
        Ω_true = F' * inv(Σ) * F
        λ, Ω = GeneralisedFilters.natural_params(pl)

        @test λ ≈ λ_true
        @test Ω ≈ Ω_true atol = 1e-6  # slight numerical differences due to jitter
    end
end

@testitem "Backward information predictor non-homogeneous" begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs

    SEED = 1234
    Dx = 2
    Dy = 2
    T = 4

    rng = StableRNG(SEED)
    model = GeneralisedFilters.GFTest.create_nonhomogeneous_linear_gaussian_model(
        rng, Dx, Dy, T
    )
    _, _, ys = simulate(rng, model, T)

    res = GeneralisedFilters.resolve
    BIF = BackwardInformationPredictor(; initial_jitter=1e-8)
    pl = backward_initialise(BIF, res(model.obs, (; t=T)), ys[T])
    pl = backward_predict(BIF, pl, res(model.dyn, (; t=T)))
    pl = backward_update(BIF, pl, res(model.obs, (; t=T - 1)), ys[T - 1])
    pl = backward_predict(BIF, pl, res(model.dyn, (; t=T - 1)))
    pl = backward_update(BIF, pl, res(model.obs, (; t=T - 2)), ys[T - 2])

    # Compute analytical result with time-varying parameters
    A_Tm1, b_Tm1, Q_Tm1 = let d = res(model.dyn, (; t=T - 1))
        d.A, d.b, d.Q
    end
    A_T, b_T, Q_T = let d = res(model.dyn, (; t=T))
        d.A, d.b, d.Q
    end
    H_Tm2, c_Tm2, R_Tm2 = let o = res(model.obs, (; t=T - 2))
        o.H, o.c, o.R
    end
    H_Tm1, c_Tm1, R_Tm1 = let o = res(model.obs, (; t=T - 1))
        o.H, o.c, o.R
    end
    H_T, c_T, R_T = let o = res(model.obs, (; t=T))
        o.H, o.c, o.R
    end

    # Projection matrix F from x_{T-2} to [Y_{T-2}, Y_{T-1}, Y_T]
    F = [H_Tm2; H_Tm1 * A_Tm1; H_T * A_T * A_Tm1]

    # Offset vector g
    g = [c_Tm2; H_Tm1 * b_Tm1 + c_Tm1; H_T * A_T * b_Tm1 + H_T * b_T + c_T]

    # Covariance Σ of [Y_{T-2}, Y_{T-1}, Y_T] given x_{T-2}
    #! format: off
    Σ = [
        R_Tm2            zeros(Dy, Dy)                     zeros(Dy, Dy);
        zeros(Dy, Dy)    H_Tm1 * Q_Tm1 * H_Tm1' + R_Tm1    H_Tm1 * Q_Tm1 * A_T' * H_T';
        zeros(Dy, Dy)    H_T * A_T * Q_Tm1 * H_Tm1'        H_T * A_T * Q_Tm1 * A_T' * H_T' + H_T * Q_T * H_T' + R_T
    ]
    #! format: on

    λ_true = F' * inv(Σ) * (vcat(ys[(T - 2):T]...) .- g)
    Ω_true = F' * inv(Σ) * F
    λ, Ω = GeneralisedFilters.natural_params(pl)

    @test λ ≈ λ_true
    @test Ω ≈ Ω_true atol = 1e-6
end

## RTS Smoothing ############################################################################

@testitem "Kalman smoother" begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs

    SEED = 1234
    Dx = 3
    Dys = [2, 3, 4]
    T = 2

    for Dy in Dys
        rng = StableRNG(SEED)
        model = GeneralisedFilters.GFTest.create_linear_gaussian_model(
            rng, Dx, Dy; static_arrays=true
        )
        _, _, ys = simulate(rng, model, T)

        # Forward pass: store filtered and predicted distributions
        kf = KF()
        filtered = Vector{GaussianState}(undef, T)
        predicted = Vector{GaussianState}(undef, T)

        state = initialise(rng, model.prior, kf)
        total_ll = 0.0
        for t in 1:T
            pred = predict(rng, model.dyn, kf, t, state, ys[t])
            predicted[t] = pred
            state, ll = update(model.obs, kf, t, pred, ys[t])
            filtered[t] = state
            total_ll += ll
        end

        # Backward pass using the RTS kernel; atom index t+1 parameterises x_t → x_{t+1}.
        smoothed = filtered[T]
        for t in (T - 1):-1:1
            d = GeneralisedFilters.resolve(model.dyn, (; t=t + 1))
            smoothed = GeneralisedFilters.rts_backward_step(
                filtered[t], d, smoothed, predicted[t + 1]
            )
        end

        # Compute ground truth using joint MVN conditional distribution
        # Let Z = [X0, X1, X2, Y1, Y2] be the joint state vector
        μ_Z, Σ_Z = GeneralisedFilters.GFTest._compute_joint(model, T)

        y = [ys[1]; ys[2]]
        I_x = (Dx + 1):(2Dx)  # just X1
        I_y = (3Dx + 1):(3Dx + 2Dy)  # Y1 and Y2
        μ_X1 = μ_Z[I_x] + Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ (y - μ_Z[I_y]))
        Σ_X1 = Σ_Z[I_x, I_x] - Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ Σ_Z[I_y, I_x])

        @test smoothed.μ ≈ μ_X1
        @test smoothed.Σ ≈ Σ_X1
    end
end

@testitem "Kalman smoother non-homogeneous" begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs

    SEED = 1234
    Dx = 2
    Dy = 2
    T = 3

    rng = StableRNG(SEED)
    model = GeneralisedFilters.GFTest.create_nonhomogeneous_linear_gaussian_model(
        rng, Dx, Dy, T
    )
    _, _, ys = simulate(rng, model, T)

    kf = KF()
    filtered = Vector{GaussianState}(undef, T)
    predicted = Vector{GaussianState}(undef, T)

    let state = initialise(rng, model.prior, kf)
        for t in 1:T
            pred = predict(rng, model.dyn, kf, t, state, ys[t])
            predicted[t] = pred
            state, _ = update(model.obs, kf, t, pred, ys[t])
            filtered[t] = state
        end
    end

    smoothed = foldl((T - 1):-1:1; init=filtered[T]) do smoothed, t
        d = GeneralisedFilters.resolve(model.dyn, (; t=t + 1))
        GeneralisedFilters.rts_backward_step(filtered[t], d, smoothed, predicted[t + 1])
    end

    μ_Z, Σ_Z = GeneralisedFilters.GFTest._compute_joint_nonhomogeneous(model, T)

    y = vcat(ys...)
    I_x = (Dx + 1):(2Dx)
    I_y = (Dx * (T + 1) + 1):(Dx * (T + 1) + T * Dy)
    μ_X1 = μ_Z[I_x] + Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ (y - μ_Z[I_y]))
    Σ_X1 = Σ_Z[I_x, I_x] - Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ Σ_Z[I_y, I_x])

    @test smoothed.μ ≈ μ_X1
    @test smoothed.Σ ≈ Σ_X1
end

@testitem "RTS smoother predicted cache" begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs

    SEED = 1234
    Dx = 3
    Dy = 2
    T = 5

    rng = StableRNG(SEED)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(
        rng, Dx, Dy; static_arrays=true
    )
    _, _, ys = simulate(rng, model, T)

    kf = KF()
    filtered = Vector{GaussianState}(undef, T)
    predicted = Vector{GaussianState}(undef, T)

    let state = initialise(rng, model.prior, kf)
        for t in 1:T
            pred = predict(rng, model.dyn, kf, t, state, ys[t])
            predicted[t] = pred
            state, _ = update(model.obs, kf, t, pred, ys[t])
            filtered[t] = state
        end
    end

    res = GeneralisedFilters.resolve
    rts = GeneralisedFilters.rts_backward_step

    # Smooth with predicted provided
    smoothed_with_pred = foldl((T - 1):-1:1; init=filtered[T]) do smoothed, t
        rts(filtered[t], res(model.dyn, (; t=t + 1)), smoothed, predicted[t + 1])
    end

    # Smooth without predicted (recomputed internally)
    smoothed_without_pred = foldl((T - 1):-1:1; init=filtered[T]) do smoothed, t
        rts(filtered[t], res(model.dyn, (; t=t + 1)), smoothed)
    end

    @test smoothed_with_pred.μ ≈ smoothed_without_pred.μ
    @test smoothed_with_pred.Σ ≈ smoothed_without_pred.Σ
end

## Two-Filter Smoothing #####################################################################

@testitem "Kalman two-filter smoother" begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs

    SEED = 1234
    Dx = 3
    Dy = 2
    T = 5
    t_smooth = 2

    rng = StableRNG(SEED)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(
        rng, Dx, Dy; static_arrays=true
    )
    _, _, ys = simulate(rng, model, T)

    kf = KF()
    filtered = Vector{GaussianState}(undef, T)

    let state = initialise(rng, model.prior, kf)
        for t in 1:T
            pred = predict(rng, model.dyn, kf, t, state, ys[t])
            state, _ = update(model.obs, kf, t, pred, ys[t])
            filtered[t] = state
        end
    end

    res = GeneralisedFilters.resolve

    # Backward information pass: compute p(y_{t_smooth+1:T} | x_{t_smooth}).
    # initial_jitter needed because Dy < Dx makes H'R⁻¹H rank-deficient.
    bip = BackwardInformationPredictor(; initial_jitter=1e-10)
    back_lik = let lik = backward_initialise(bip, res(model.obs, (; t=T)), ys[T])
        for t in (T - 1):-1:(t_smooth + 1)
            lik = backward_predict(bip, lik, res(model.dyn, (; t=t + 1)))
            lik = backward_update(bip, lik, res(model.obs, (; t)), ys[t])
        end
        # Final predict at t_smooth: transition t_smooth → t_smooth+1.
        backward_predict(bip, lik, res(model.dyn, (; t=t_smooth + 1)))
    end

    smoothed_2f = two_filter_smooth(filtered[t_smooth], back_lik)

    # Compare to RTS smoother result
    rts = GeneralisedFilters.rts_backward_step
    smoothed_rts = foldl((T - 1):-1:t_smooth; init=filtered[T]) do smoothed, t
        rts(filtered[t], res(model.dyn, (; t=t + 1)), smoothed)
    end

    @test smoothed_2f.μ ≈ smoothed_rts.μ
    @test smoothed_2f.Σ ≈ smoothed_rts.Σ
end
