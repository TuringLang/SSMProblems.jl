"""Unit tests for Kalman filter and smoother algorithms."""

## Forward Filtering ########################################################################

@testitem "Kalman filter test" begin
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
        _, _, ys = sample(rng, model, 1)

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

@testitem "Kalman filter StaticArrays test" begin
    using GeneralisedFilters
    using SSMProblems
    using StableRNGs
    using StaticArrays
    using PDMats

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

    model = create_homogeneous_linear_gaussian_model(
        μ0, PDMat(Σ0), A, b, PDMat(Q), H, c, PDMat(R)
    )

    _, _, ys = sample(rng, model, 2)

    state, _ = GeneralisedFilters.filter(rng, model, KalmanFilter(), ys)

    # Verify returned values are still StaticArrays
    @test ys[2] isa SVector{D,Float64}
    @test state.μ isa SVector{D,Float64}
    @test state.Σ isa PDMat{Float64,SMatrix{D,D,Float64,D * D}}
end

## Backward Information Filtering ###########################################################

@testitem "Backward information predictor test" begin
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
        _, _, ys = sample(rng, model, T)

        # Perform backward information filtering
        # Need initial jitter when Dy < Dx to ensure PD covariance
        BIF = BackwardInformationPredictor(; initial_jitter=1e-8)
        predictive_likelihood = backward_initialise(rng, model.obs, BIF, T, ys[T])
        predictive_likelihood = backward_predict(
            rng, model.dyn, BIF, T - 1, predictive_likelihood
        )
        predictive_likelihood = backward_update(
            model.obs, BIF, T - 1, predictive_likelihood, ys[T - 1]
        )
        predictive_likelihood = backward_predict(
            rng, model.dyn, BIF, T - 2, predictive_likelihood
        )
        predictive_likelihood = backward_update(
            model.obs, BIF, T - 2, predictive_likelihood, ys[T - 2]
        )

        # Assuming homogenous
        A, b, Q = calc_params(model.dyn, T - 1)
        H, c, R = calc_params(model.obs, T;)
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
        λ, Ω = GeneralisedFilters.natural_params(predictive_likelihood)

        @test λ ≈ λ_true
        @test Ω ≈ Ω_true atol = 1e-6  # slight numerical differences due to jitter
    end
end

@testitem "Backward information predictor non-homogeneous test" begin
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
    _, _, ys = sample(rng, model, T)

    BIF = BackwardInformationPredictor(; initial_jitter=1e-8)
    predictive_likelihood = backward_initialise(rng, model.obs, BIF, T, ys[T])
    predictive_likelihood = backward_predict(
        rng, model.dyn, BIF, T - 1, predictive_likelihood
    )
    predictive_likelihood = backward_update(
        model.obs, BIF, T - 1, predictive_likelihood, ys[T - 1]
    )
    predictive_likelihood = backward_predict(
        rng, model.dyn, BIF, T - 2, predictive_likelihood
    )
    predictive_likelihood = backward_update(
        model.obs, BIF, T - 2, predictive_likelihood, ys[T - 2]
    )

    # Compute analytical result with time-varying parameters
    A_Tm1, b_Tm1, Q_Tm1 = calc_params(model.dyn, T - 1)
    A_T, b_T, Q_T = calc_params(model.dyn, T)
    H_Tm2, c_Tm2, R_Tm2 = calc_params(model.obs, T - 2)
    H_Tm1, c_Tm1, R_Tm1 = calc_params(model.obs, T - 1)
    H_T, c_T, R_T = calc_params(model.obs, T)

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
    λ, Ω = GeneralisedFilters.natural_params(predictive_likelihood)

    @test λ ≈ λ_true
    @test Ω ≈ Ω_true atol = 1e-6
end

## RTS Smoothing ############################################################################

@testitem "Kalman smoother test" begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs
    using SSMProblems: dyn, obs, prior

    SEED = 1234
    Dx = 3
    Dys = [2, 3, 4]
    T = 2

    for Dy in Dys
        rng = StableRNG(SEED)
        model = GeneralisedFilters.GFTest.create_linear_gaussian_model(
            rng, Dx, Dy; static_arrays=true
        )
        _, _, ys = sample(rng, model, T)

        # Forward pass: store filtered and predicted distributions
        kf = KF()
        filtered = Vector{MvNormal}(undef, T)
        predicted = Vector{MvNormal}(undef, T)

        state = initialise(rng, prior(model), kf)
        total_ll = 0.0
        for t in 1:T
            pred = predict(rng, dyn(model), kf, t, state, ys[t])
            predicted[t] = pred
            state, ll = update(obs(model), kf, t, pred, ys[t])
            filtered[t] = state
            total_ll += ll
        end

        # Backward pass: smooth using backward_smooth
        smoothed = filtered[T]
        for t in (T - 1):-1:1
            smoothed = backward_smooth(
                dyn(model), kf, t, filtered[t], smoothed; predicted=predicted[t + 1]
            )
        end

        # Compute ground truth using joint MVN conditional distribution
        # Let Z = [X0, X1, X2, Y1, Y2] be the joint state vector
        μ_Z, Σ_Z = GeneralisedFilters.GFTest._compute_joint(model, T)

        # Condition on observations using formula for MVN conditional distribution. See:
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        y = [ys[1]; ys[2]]
        I_x = (Dx + 1):(2Dx)  # just X1
        I_y = (3Dx + 1):(3Dx + 2Dy)  # Y1 and Y2
        μ_X1 = μ_Z[I_x] + Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ (y - μ_Z[I_y]))
        Σ_X1 = Σ_Z[I_x, I_x] - Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ Σ_Z[I_y, I_x])

        @test smoothed.μ ≈ μ_X1
        @test smoothed.Σ ≈ Σ_X1
    end
end

@testitem "Kalman smoother non-homogeneous test" begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs
    using SSMProblems: dyn, obs, prior

    SEED = 1234
    Dx = 2
    Dy = 2
    T = 3

    rng = StableRNG(SEED)
    model = GeneralisedFilters.GFTest.create_nonhomogeneous_linear_gaussian_model(
        rng, Dx, Dy, T
    )
    _, _, ys = sample(rng, model, T)

    # Forward pass: store filtered and predicted distributions
    kf = KF()
    filtered = Vector{MvNormal}(undef, T)
    predicted = Vector{MvNormal}(undef, T)

    let state = initialise(rng, prior(model), kf)
        for t in 1:T
            pred = predict(rng, dyn(model), kf, t, state, ys[t])
            predicted[t] = pred
            state, _ = update(obs(model), kf, t, pred, ys[t])
            filtered[t] = state
        end
    end

    # Backward pass: smooth using backward_smooth
    smoothed = foldl((T - 1):-1:1; init=filtered[T]) do smoothed, t
        backward_smooth(dyn(model), kf, t, filtered[t], smoothed; predicted=predicted[t + 1])
    end

    # Compute ground truth using joint MVN conditional distribution
    μ_Z, Σ_Z = GeneralisedFilters.GFTest._compute_joint_nonhomogeneous(model, T)

    # Condition on observations
    y = vcat(ys...)
    I_x = (Dx + 1):(2Dx)
    I_y = (Dx * (T + 1) + 1):(Dx * (T + 1) + T * Dy)
    μ_X1 = μ_Z[I_x] + Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ (y - μ_Z[I_y]))
    Σ_X1 = Σ_Z[I_x, I_x] - Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ Σ_Z[I_y, I_x])

    @test smoothed.μ ≈ μ_X1
    @test smoothed.Σ ≈ Σ_X1
end

@testitem "RTS smoother predicted cache test" begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs
    using SSMProblems: dyn, obs, prior

    SEED = 1234
    Dx = 3
    Dy = 2
    T = 5

    rng = StableRNG(SEED)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(
        rng, Dx, Dy; static_arrays=true
    )
    _, _, ys = sample(rng, model, T)

    # Forward pass: store filtered and predicted distributions
    kf = KF()
    filtered = Vector{MvNormal}(undef, T)
    predicted = Vector{MvNormal}(undef, T)

    state = let s = initialise(rng, prior(model), kf)
        for t in 1:T
            pred = predict(rng, dyn(model), kf, t, s, ys[t])
            predicted[t] = pred
            s, _ = update(obs(model), kf, t, pred, ys[t])
            filtered[t] = s
        end
        s
    end

    # Smooth with predicted provided
    smoothed_with_pred = foldl((T - 1):-1:1; init=filtered[T]) do smoothed, t
        backward_smooth(dyn(model), kf, t, filtered[t], smoothed; predicted=predicted[t + 1])
    end

    # Smooth without predicted (computed internally)
    smoothed_without_pred = foldl((T - 1):-1:1; init=filtered[T]) do smoothed, t
        backward_smooth(dyn(model), kf, t, filtered[t], smoothed)
    end

    # Both should give the same result
    @test smoothed_with_pred.μ ≈ smoothed_without_pred.μ
    @test smoothed_with_pred.Σ ≈ smoothed_without_pred.Σ
end

## Two-Filter Smoothing #####################################################################

@testitem "Kalman two-filter smoother test" begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs
    using SSMProblems: dyn, obs, prior

    SEED = 1234
    Dx = 3
    Dy = 2
    T = 5
    t_smooth = 2

    rng = StableRNG(SEED)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(
        rng, Dx, Dy; static_arrays=true
    )
    _, _, ys = sample(rng, model, T)

    # Forward pass: store filtered distributions
    kf = KF()
    filtered = Vector{MvNormal}(undef, T)

    let state = initialise(rng, prior(model), kf)
        for t in 1:T
            pred = predict(rng, dyn(model), kf, t, state, ys[t])
            state, _ = update(obs(model), kf, t, pred, ys[t])
            filtered[t] = state
        end
    end

    # Backward information pass: compute p(y_{t_smooth+1:T} | x_{t_smooth})
    # We do predict+update from T-1 down to t_smooth+1, then only predict at t_smooth
    # Note: initial_jitter needed because Dy < Dx makes H'R⁻¹H rank-deficient
    bip = BackwardInformationPredictor(; initial_jitter=1e-10)
    back_lik = let lik = backward_initialise(rng, obs(model), bip, T, ys[T])
        for t in (T - 1):-1:(t_smooth + 1)
            lik = backward_predict(rng, dyn(model), bip, t, lik)
            lik = backward_update(obs(model), bip, t, lik, ys[t])
        end
        # Final predict at t_smooth (no update - we don't include y_{t_smooth} in backward lik)
        backward_predict(rng, dyn(model), bip, t_smooth, lik)
    end

    # Two-filter smooth at t_smooth
    smoothed_2f = two_filter_smooth(filtered[t_smooth], back_lik)

    # Compare to RTS smoother result
    smoothed_rts = foldl((T - 1):-1:t_smooth; init=filtered[T]) do smoothed, t
        backward_smooth(dyn(model), kf, t, filtered[t], smoothed)
    end

    @test smoothed_2f.μ ≈ smoothed_rts.μ
    @test smoothed_2f.Σ ≈ smoothed_rts.Σ
end
