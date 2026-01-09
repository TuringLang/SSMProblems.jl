using Test
using TestItems
using TestItemRunner

@run_package_tests filter = ti -> !(:gpu in ti.tags)

# include("Aqua.jl")
# include("type_stability.jl")
# include("resamplers.jl")
# include("kalman_gradient.jl")

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

@testitem "Kalman filter StaticArrays" begin
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

@testitem "RTS smoother (single cache version)" begin
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

@testitem "Bootstrap filter test" begin
    using SSMProblems
    using StableRNGs
    using StatsBase: weights

    rng = StableRNG(1234)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(
        rng, 1, 1; static_arrays=true
    )
    _, _, ys = sample(rng, model, 4)

    resampler = GeneralisedFilters.GFTest.AlternatingResampler()
    bf = BF(10^6; resampler=resampler)
    bf_state, llbf = GeneralisedFilters.filter(rng, model, bf, ys)
    kf_state, llkf = GeneralisedFilters.filter(rng, model, KF(), ys)

    xs = getfield.(bf_state.particles, :state)
    ws = weights(bf_state)

    # Compare log-likelihood and states
    @test first(kf_state.μ) ≈ sum(first.(xs) .* ws) rtol = 1e-3
    @test llkf ≈ llbf atol = 1e-3
end

@testitem "Guided filter test" begin
    using SSMProblems
    using StableRNGs
    using StatsBase: weights

    rng = StableRNG(1234)
    model = GeneralisedFilters.GFTest.create_linear_gaussian_model(
        rng, 1, 1; static_arrays=true
    )
    _, _, ys = sample(rng, model, 4)

    prop = GeneralisedFilters.GFTest.OptimalProposal(model.dyn, model.obs)
    resampler = GeneralisedFilters.GFTest.AlternatingResampler()
    gf = ParticleFilter(10^6, prop; resampler=resampler)
    gf_state, llgf = GeneralisedFilters.filter(rng, model, gf, ys)
    kf_state, llkf = GeneralisedFilters.filter(rng, model, KF(), ys)

    xs = getfield.(gf_state.particles, :state)
    ws = weights(gf_state)

    @test first(kf_state.μ) ≈ sum(first.(xs) .* ws) rtol = 1e-3
    @test llkf ≈ llgf atol = 1e-3
end

@testitem "Discrete filter test" begin
    using GeneralisedFilters
    using Distributions
    using StableRNGs
    using SSMProblems

    rng = StableRNG(1234)
    α0 = rand(rng, 3)
    α0 = α0 / sum(α0)
    P = rand(rng, 3, 3)
    P = P ./ sum(P; dims=2)

    struct MixtureModelObservation{T<:Real,MT<:AbstractVector{T}} <: ObservationProcess
        μs::MT
    end

    function SSMProblems.logdensity(
        obs::MixtureModelObservation{T},
        step::Integer,
        state::Integer,
        observation;
        kwargs...,
    ) where {T}
        return logpdf(Normal(obs.μs[state], one(T)), observation)
    end

    μs = [0.0, 1.0, 2.0]

    prior = HomogeneousDiscretePrior(α0)
    dyn = HomogeneousDiscreteLatentDynamics(P)
    obs = MixtureModelObservation(μs)
    model = StateSpaceModel(prior, dyn, obs)

    observations = [rand(rng)]

    df = DiscreteFilter()
    state, ll = GeneralisedFilters.filter(model, df, observations)

    # Brute force calculations of each conditional path probability p(x_{1:T} | y_{1:T})
    T = 1
    K = 3
    y = only(observations)
    path_probs = Dict{Tuple{Int,Int},Float64}()
    for x0 in 1:K, x1 in 1:K
        prior_prob = α0[x0] * P[x0, x1]
        likelihood = exp(SSMProblems.logdensity(obs, 1, x1, y))
        path_probs[(x0, x1)] = prior_prob * likelihood
    end
    marginal = sum(values(path_probs))

    filtered_paths = Base.filter(((k, v),) -> k[end] == 1, path_probs)
    @test state[1] ≈ sum(values(filtered_paths)) / marginal
    @test ll ≈ log(marginal)
end

@testitem "Backward discrete predictor test" begin
    using GeneralisedFilters
    using Distributions
    using Random
    using SSMProblems
    using LogExpFunctions

    # Simple 3-state HMM with Gaussian emissions
    K = 3
    T = 4

    α0 = [0.5, 0.3, 0.2]
    P = [
        0.7 0.2 0.1
        0.1 0.8 0.1
        0.2 0.2 0.6
    ]

    struct BackwardTestObservation{T<:Real,MT<:AbstractVector{T}} <: ObservationProcess
        μs::MT
    end

    function SSMProblems.logdensity(
        obs::BackwardTestObservation{T},
        step::Integer,
        state::Integer,
        observation;
        kwargs...,
    ) where {T}
        return logpdf(Normal(obs.μs[state], one(T)), observation)
    end

    μs = [0.0, 2.0, 4.0]
    obs = BackwardTestObservation(μs)

    observations = [0.5, 1.8, 3.5, 2.1]

    # Run backward predictor
    algo = BackwardDiscretePredictor()
    rng = Random.default_rng()
    dyn = HomogeneousDiscreteLatentDynamics(P)

    # Initialize at time T and run backward pass
    β = (
        let lik = GeneralisedFilters.backward_initialise(
                rng, obs, algo, T, observations[T]; num_states=K
            )
            for t in (T - 1):-1:1
                lik = GeneralisedFilters.backward_predict(rng, dyn, algo, t, lik)
                lik = GeneralisedFilters.backward_update(obs, algo, t, lik, observations[t])
            end
            lik
        end
    )

    # Brute force: compute β_1(i) = p(y_{1:T} | x_1 = i) by enumerating all paths
    log_β_bruteforce = zeros(K)
    for x1 in 1:K
        log_prob = -Inf
        for x2 in 1:K, x3 in 1:K, x4 in 1:K
            log_path_prob = 0.0
            # Transitions
            log_path_prob += log(P[x1, x2]) + log(P[x2, x3]) + log(P[x3, x4])
            # Emissions
            log_path_prob += logpdf(Normal(μs[x1], 1.0), observations[1])
            log_path_prob += logpdf(Normal(μs[x2], 1.0), observations[2])
            log_path_prob += logpdf(Normal(μs[x3], 1.0), observations[3])
            log_path_prob += logpdf(Normal(μs[x4], 1.0), observations[4])
            log_prob = logaddexp(log_prob, log_path_prob)
        end
        log_β_bruteforce[x1] = log_prob
    end

    @test log_likelihoods(β) ≈ log_β_bruteforce
end

@testitem "Discrete smoother test" begin
    using GeneralisedFilters
    using Distributions
    using Random
    using SSMProblems
    using LogExpFunctions

    K = 3
    T = 4
    t_smooth = 2

    α0 = [0.5, 0.3, 0.2]
    P = [
        0.7 0.2 0.1
        0.1 0.8 0.1
        0.2 0.2 0.6
    ]

    struct SmootherTestObservation{T<:Real,MT<:AbstractVector{T}} <: ObservationProcess
        μs::MT
    end

    function SSMProblems.logdensity(
        obs::SmootherTestObservation{T},
        step::Integer,
        state::Integer,
        observation;
        kwargs...,
    ) where {T}
        return logpdf(Normal(obs.μs[state], one(T)), observation)
    end

    μs = [0.0, 2.0, 4.0]
    obs = SmootherTestObservation(μs)

    prior = HomogeneousDiscretePrior(α0)
    dyn = HomogeneousDiscreteLatentDynamics(P)
    model = StateSpaceModel(prior, dyn, obs)

    observations = [0.5, 1.8, 3.5, 2.1]

    rng = Random.default_rng()
    smoothed, ll = smooth(rng, model, DiscreteSmoother(), observations; t_smooth=t_smooth)

    # Brute force: compute γ_{t_smooth}(i) = p(x_{t_smooth} = i | y_{1:T})
    # by enumerating all paths and marginalizing
    log_joint_probs = Dict{NTuple{5,Int},Float64}()
    for x0 in 1:K, x1 in 1:K, x2 in 1:K, x3 in 1:K, x4 in 1:K
        log_prob = 0.0
        log_prob += log(α0[x0])
        log_prob += log(P[x0, x1]) + log(P[x1, x2]) + log(P[x2, x3]) + log(P[x3, x4])
        log_prob += logpdf(Normal(μs[x1], 1.0), observations[1])
        log_prob += logpdf(Normal(μs[x2], 1.0), observations[2])
        log_prob += logpdf(Normal(μs[x3], 1.0), observations[3])
        log_prob += logpdf(Normal(μs[x4], 1.0), observations[4])
        log_joint_probs[(x0, x1, x2, x3, x4)] = log_prob
    end

    log_marginal = logsumexp(collect(values(log_joint_probs)))

    # Marginalize to get p(x_{t_smooth} | y_{1:T})
    # t_smooth=2 corresponds to x2 (index 3 in the tuple since x0 is index 1)
    γ_bruteforce = zeros(K)
    for i in 1:K
        matching_paths = Base.filter(((k, v),) -> k[t_smooth + 1] == i, log_joint_probs)
        if !isempty(matching_paths)
            γ_bruteforce[i] = exp(logsumexp(collect(values(matching_paths))) - log_marginal)
        end
    end

    @test smoothed ≈ γ_bruteforce
    @test ll ≈ log_marginal
end

# @testitem "Rao-Blackwellised BF test" begin
#     using Distributions
#     using GeneralisedFilters
#     using LinearAlgebra
#     using SSMProblems
#     using StableRNGs
#     using StatsBase: weights

#     rng = StableRNG(1234)

#     full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
#         rng, 1, 1, 1; static_arrays=true
#     )
#     _, _, ys = sample(rng, full_model, 4)

#     resampler = GeneralisedFilters.GFTest.AlternatingResampler()
#     bf = BF(10^6; resampler=resampler)
#     rbbf = RBPF(bf, KalmanFilter())

#     rbbf_state, llrbbf = GeneralisedFilters.filter(rng, hier_model, rbbf, ys)
#     xs = getfield.(getfield.(rbbf_state.particles, :state), :x)
#     zs = getfield.(getfield.(rbbf_state.particles, :state), :z)
#     ws = weights(rbbf_state)

#     kf_state, llkf = GeneralisedFilters.filter(rng, full_model, KF(), ys)

#     @test first(kf_state.μ) ≈ sum(only.(xs) .* ws) rtol = 1e-3
#     @test last(kf_state.μ) ≈ sum(only.(getfield.(zs, :μ)) .* ws) rtol = 1e-3
#     @test llkf ≈ llrbbf atol = 1e-3
# end

# @testitem "Rao-Blackwellised GF test" begin
#     using SSMProblems
#     using StableRNGs
#     using StatsBase: weights

#     rng = StableRNG(1234)

#     full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
#         rng, 1, 1, 1; static_arrays=true
#     )
#     _, _, ys = sample(rng, full_model, 4)

#     prop = GeneralisedFilters.GFTest.OverdispersedProposal(dyn(hier_model).outer_dyn, 1.5)
#     resampler = GeneralisedFilters.GFTest.AlternatingResampler()
#     gf = ParticleFilter(10^6, prop; resampler=resampler)
#     rbgf = RBPF(gf, KalmanFilter())
#     rbgf_state, llrbgf = GeneralisedFilters.filter(rng, hier_model, rbgf, ys)
#     xs = getfield.(getfield.(rbgf_state.particles, :state), :x)
#     zs = getfield.(getfield.(rbgf_state.particles, :state), :z)
#     ws = weights(rbgf_state)

#     kf_state, llkf = GeneralisedFilters.filter(rng, full_model, KF(), ys)

#     @test first(kf_state.μ) ≈ sum(only.(xs) .* ws) rtol = 1e-3
#     @test last(kf_state.μ) ≈ sum(only.(getfield.(zs, :μ)) .* ws) rtol = 1e-3
#     @test llkf ≈ llrbgf atol = 1e-3
# end

# @testitem "ABF test" begin
#     using Distributions
#     using GeneralisedFilters
#     using LinearAlgebra
#     using SSMProblems
#     using StableRNGs
#     using StatsBase: weights

#     rng = StableRNG(1234)
#     model = GeneralisedFilters.GFTest.create_linear_gaussian_model(
#         rng, 1, 1; static_arrays=true
#     )
#     _, _, ys = sample(rng, model, 4)

#     # resampler = GeneralisedFilters.GFTest.AlternatingResampler()
#     resampler = ESSResampler(0.8)
#     bf = BF(10^6; resampler=resampler)
#     abf = AuxiliaryParticleFilter(bf, MeanPredictive())
#     abf_state, llabf = GeneralisedFilters.filter(rng, model, abf, ys)
#     kf_state, llkf = GeneralisedFilters.filter(rng, model, KF(), ys)

#     xs = getfield.(abf_state.particles, :state)
#     ws = weights(abf_state)

#     @test first(kf_state.μ) ≈ sum(first.(xs) .* ws) rtol = 1e-2
#     @test llkf ≈ llabf atol = 1e-3
# end

# @testitem "ARBF test" begin
#     using Distributions
#     using GeneralisedFilters
#     using LinearAlgebra
#     using SSMProblems
#     using StableRNGs
#     using StatsBase: weights

#     rng = StableRNG(1234)

#     full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
#         rng, 1, 1, 1; static_arrays=true
#     )
#     _, _, ys = sample(rng, hier_model, 4)

#     resampler = GeneralisedFilters.GFTest.AlternatingResampler()
#     bf = BF(10^6; resampler=resampler)
#     rbbf = RBPF(bf, KalmanFilter())
#     arbf = AuxiliaryParticleFilter(rbbf, MeanPredictive())
#     arbf_state, llarbf = GeneralisedFilters.filter(rng, hier_model, arbf, ys)
#     xs = getfield.(getfield.(arbf_state.particles, :state), :x)
#     zs = getfield.(getfield.(arbf_state.particles, :state), :z)
#     ws = weights(arbf_state)

#     kf_state, llkf = GeneralisedFilters.filter(rng, full_model, KF(), ys)

#     @test first(kf_state.μ) ≈ sum(only.(xs) .* ws) rtol = 1e-2
#     @test last(kf_state.μ) ≈ sum(only.(getfield.(zs, :μ)) .* ws) rtol = 1e-3
#     @test llkf ≈ llarbf atol = 1e-3
# end

# @testitem "RBPF ancestory test" begin
#     using SSMProblems
#     using StableRNGs

#     SEED = 1234
#     T = 5
#     N_particles = 100

#     rng = StableRNG(SEED)
#     full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
#         rng, 1, 1, 1
#     )
#     _, _, ys = sample(rng, full_model, T)

#     cb = GeneralisedFilters.AncestorCallback(nothing)
#     rbpf = RBPF(BF(N_particles; threshold=0.8), KalmanFilter())
#     GeneralisedFilters.filter(rng, hier_model, rbpf, ys; callback=cb)

#     # TODO: add proper test comparing to dense storage
#     tree = cb.tree
#     paths = GeneralisedFilters.get_ancestry(tree)
# end

# @testitem "BF on hierarchical model test" begin
#     using SSMProblems
#     using StableRNGs
#     using StatsBase: weights

#     SEED = 1234
#     D_outer = 1
#     D_inner = 1
#     D_obs = 1
#     T = 5
#     N_particles = 10^4

#     rng = StableRNG(SEED)

#     full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
#         rng, D_outer, D_inner, D_obs
#     )
#     _, _, ys = sample(rng, full_model, T)

#     # Ground truth Kalman filtering
#     kf_states, kf_ll = GeneralisedFilters.filter(rng, full_model, KalmanFilter(), ys)

#     # Rao-Blackwellised particle filtering
#     bf = BF(N_particles; threshold=0.8)
#     states, ll = GeneralisedFilters.filter(rng, hier_model, bf, ys)

#     # Extract final filtered states
#     xs = map(p -> getproperty(p.state, :x), states.particles)
#     zs = map(p -> getproperty(p.state, :z), states.particles)
#     ws = weights(states)

#     @test kf_ll ≈ ll rtol = 1e-2

#     # Higher tolerance for outer state since variance is higher
#     @test first(kf_states.μ) ≈ sum(only.(xs) .* ws) rtol = 1e-1
#     @test last(kf_states.μ) ≈ sum(only.(zs) .* ws) rtol = 1e-1
# end

# @testitem "Dense ancestry test" begin
#     using GeneralisedFilters
#     using StableRNGs
#     using PDMats
#     using LinearAlgebra
#     using Random: randexp, AbstractRNG
#     using StatsBase: sample, Weights

#     using OffsetArrays

#     struct DummyResampler <: GeneralisedFilters.AbstractResampler end

#     function GeneralisedFilters.sample_ancestors(
#         ::AbstractRNG, ::DummyResampler, weights::AbstractVector, n::Int64=length(weights)
#     )
#         return [mod1(a - 1, length(weights)) for a in 1:n]
#     end

#     SEED = 1234
#     K = 5
#     N_particles = max(10, K + 2)

#     rng = StableRNG(SEED)
#     model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, 1, 1)
#     _, _, ys = sample(rng, model, K)

#     ref_traj = OffsetVector([rand(rng, 1) for _ in 0:K], -1)

#     bf = BF(N_particles; threshold=1.0, resampler=DummyResampler())
#     cb = GeneralisedFilters.DenseAncestorCallback(nothing)
#     bf_state, _ = GeneralisedFilters.filter(
#         rng, model, bf, ys; ref_state=ref_traj, callback=cb
#     )

#     traj = GeneralisedFilters.get_ancestry(cb.container, N_particles)
#     true_traj = [cb.container.particles[t][N_particles - K + t] for t in 0:K]

#     @test traj.parent == true_traj
#     @test GeneralisedFilters.get_ancestry(cb.container, 1) == ref_traj
# end

# @testitem "CSMC test" begin
#     using GeneralisedFilters
#     using StableRNGs
#     using PDMats
#     using LinearAlgebra
#     using LogExpFunctions: logsumexp
#     using Random: randexp
#     using StatsBase: sample, weights

#     using OffsetArrays

#     SEED = 1234
#     Dx = 1
#     Dy = 1
#     K = 10
#     t_smooth = 2
#     T = Float64
#     N_particles = 10  # Use small particle number so impact of ref state is significant
#     N_burnin = 1000
#     N_sample = 100000

#     rng = StableRNG(SEED)
#     model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)
#     _, _, ys = sample(rng, model, K)

#     # Kalman smoother
#     state, ks_ll = GeneralisedFilters.smooth(
#         rng, model, KalmanSmoother(), ys; t_smooth=t_smooth
#     )

#     N_steps = N_burnin + N_sample
#     bf = BF(N_particles; threshold=0.6)
#     ref_traj = nothing
#     trajectory_samples = []
#     lls = []

#     for i in 1:N_steps
#         cb = GeneralisedFilters.DenseAncestorCallback(nothing)
#         bf_state, ll = GeneralisedFilters.filter(
#             rng, model, bf, ys; ref_state=ref_traj, callback=cb
#         )
#         ws = weights(bf_state)
#         sampled_idx = sample(rng, 1:N_particles, ws)
#         global ref_traj = GeneralisedFilters.get_ancestry(cb.container, sampled_idx)
#         if i > N_burnin
#             push!(trajectory_samples, ref_traj)
#             push!(lls, ll)
#         end
#     end

#     # The CSMC estimate of the evidence Z = p(y_{1:T}) is biased but 1 / ̂Z is actually an
#     # unbiased estimate of 1 / Z. See Elements of Sequential Monte Carlo (Section 5.2)
#     log_recip_likelihood_estimate = logsumexp(-lls) - log(length(lls))

#     csmc_mean = sum(getindex.(trajectory_samples, t_smooth)) / N_sample
#     @test csmc_mean ≈ state.μ rtol = 1e-3
#     @test log_recip_likelihood_estimate ≈ -ks_ll rtol = 1e-3
# end

# @testitem "RBCSMC test" begin
#     using GeneralisedFilters
#     using StableRNGs
#     using PDMats
#     using LinearAlgebra
#     using Random: randexp
#     using StatsBase: sample, weights
#     using StaticArrays
#     using Statistics

#     using OffsetArrays

#     SEED = 1234
#     D_outer = 1
#     D_inner = 1
#     D_obs = 1
#     K = 5
#     t_smooth = 2
#     T = Float64
#     N_particles = 10  # Use small particle number so impact of ref state is significant
#     N_burnin = 1000
#     N_sample = 10000

#     rng = StableRNG(SEED)
#     full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
#         rng, D_outer, D_inner, D_obs, T; static_arrays=true
#     )
#     _, _, ys = sample(rng, full_model, K)
#     # Convert to static arrays
#     ys = [SVector{1,T}(y) for y in ys]

#     # Kalman smoother
#     state, _ = GeneralisedFilters.smooth(
#         rng, full_model, KalmanSmoother(), ys; t_smooth=t_smooth
#     )

#     N_steps = N_burnin + N_sample
#     rbpf = RBPF(BF(N_particles; threshold=0.8), KalmanFilter())
#     ref_traj = nothing
#     trajectory_samples = []

#     cb = GeneralisedFilters.DenseAncestorCallback(nothing)
#     for i in 1:N_steps
#         bf_state, _ = GeneralisedFilters.filter(
#             rng, hier_model, rbpf, ys; ref_state=ref_traj, callback=cb
#         )
#         ws = weights(bf_state)
#         sampled_idx = sample(rng, 1:N_particles, ws)

#         global ref_traj = GeneralisedFilters.get_ancestry(cb.container, sampled_idx)
#         if i > N_burnin
#             push!(trajectory_samples, deepcopy(ref_traj))
#         end
#         # Reference trajectory should only be nonlinear state for RBPF
#         ref_traj = getproperty.(ref_traj, :x)
#     end

#     # Extract inner and outer trajectories
#     x_trajectories = getproperty.(getindex.(trajectory_samples, t_smooth), :x)

#     # Smooth the inner (z) component using backward_smooth
#     inner_dyn = hier_model.inner_model.dyn
#     z_smoothed_means = Vector{T}(undef, N_sample)
#     for i in 1:N_sample
#         smoothed_z = trajectory_samples[i][K].z

#         for t in (K - 1):-1:t_smooth
#             filtered_z = trajectory_samples[i][t].z
#             # Pass prev_outer to condition the inner dynamics on the outer trajectory
#             smoothed_z = backward_smooth(
#                 inner_dyn,
#                 KF(),
#                 t,
#                 filtered_z,
#                 smoothed_z;
#                 prev_outer=trajectory_samples[i][t].x,
#             )
#         end

#         z_smoothed_means[i] = only(smoothed_z.μ)
#     end

#     # Compare to ground truth
#     @test state.μ[1] ≈ only(mean(x_trajectories)) rtol = 1e-2
#     @test state.μ[2] ≈ mean(z_smoothed_means) rtol = 1e-3
# end

# @testitem "RBCSMC-AS test" begin
#     using GeneralisedFilters
#     using StableRNGs
#     using PDMats
#     using LinearAlgebra
#     using Random: randexp
#     using StatsBase: sample, weights
#     using StaticArrays
#     using Statistics
#     using LogExpFunctions

#     import SSMProblems: prior, dyn, obs
#     import GeneralisedFilters: resampler, resample, move, RBState, InformationLikelihood

#     using OffsetArrays

#     SEED = 1234
#     D_outer = 1
#     D_inner = 1
#     D_obs = 1
#     K = 5
#     t_smooth = 2
#     T = Float64
#     N_particles = 10
#     N_burnin = 200
#     N_sample = 10000

#     rng = StableRNG(SEED)
#     full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
#         rng, D_outer, D_inner, D_obs, T; static_arrays=false
#     )
#     _, _, ys = sample(rng, full_model, K)

#     # Kalman smoother
#     state, _ = GeneralisedFilters.smooth(
#         rng, full_model, KalmanSmoother(), ys; t_smooth=t_smooth
#     )

#     N_steps = N_burnin + N_sample
#     rbpf = RBPF(BF(N_particles; threshold=0.8), KalmanFilter())
#     ref_traj = nothing
#     predictive_likelihoods = Vector{InformationLikelihood{Vector{T},PDMat{T,Matrix{T}}}}(
#         undef, K
#     )
#     trajectory_samples = []

#     for i in 1:N_steps
#         global predictive_likelihoods
#         cb = GeneralisedFilters.DenseAncestorCallback(nothing)

#         # Manual filtering with ancestor resampling
#         bf_state = initialise(rng, prior(hier_model), rbpf; ref_state=ref_traj)

#         # Post-Init callback
#         cb(hier_model, rbpf, bf_state, ys, PostInit)

#         for t in 1:K
#             bf_state = resample(rng, resampler(rbpf), bf_state; ref_state=ref_traj)

#             if !isnothing(ref_traj)
#                 ref_rb_state = RBState(ref_traj[t], predictive_likelihoods[t])
#                 ancestor_weights = map(bf_state.particles) do particle
#                     ancestor_weight(particle, dyn(hier_model), rbpf, t, ref_rb_state)
#                 end
#                 ancestor_idx = sample(
#                     rng, 1:N_particles, weights(softmax(ancestor_weights))
#                 )
#             end

#             bf_state, ll = move(
#                 rng, hier_model, rbpf, t, bf_state, ys[t]; ref_state=ref_traj
#             )

#             # Set ancestor index
#             if !isnothing(ref_traj)
#                 bf_state.particles[end] = GeneralisedFilters.Particle(
#                     bf_state.particles[end].state,
#                     bf_state.particles[end].log_w,
#                     ancestor_idx,
#                 )
#             end

#             # Manually trigger callback
#             cb(hier_model, rbpf, t, bf_state, ys[t], PostUpdate)
#         end

#         ws = weights(bf_state)
#         sampled_idx = sample(rng, 1:N_particles, ws)

#         global ref_traj = GeneralisedFilters.get_ancestry(cb.container, sampled_idx)
#         if i > N_burnin
#             push!(trajectory_samples, deepcopy(ref_traj))
#         end
#         # Reference trajectory should only be nonlinear state for RBPF
#         ref_traj = getproperty.(ref_traj, :x)

#         bip = BackwardInformationPredictor(; initial_jitter=1e-8)

#         pred_lik = backward_initialise(rng, hier_model.inner_model.obs, bip, K, ys[K])
#         predictive_likelihoods[K] = deepcopy(pred_lik)
#         for t in (K - 1):-1:1
#             pred_lik = backward_predict(
#                 rng,
#                 hier_model.inner_model.dyn,
#                 bip,
#                 t,
#                 pred_lik;
#                 prev_outer=ref_traj[t],
#                 new_outer=ref_traj[t + 1],
#             )
#             pred_lik = backward_update(hier_model.inner_model.obs, bip, t, pred_lik, ys[t])
#             predictive_likelihoods[t] = deepcopy(pred_lik)
#         end
#     end

#     # Extract inner and outer trajectories
#     x_trajectories = getproperty.(getindex.(trajectory_samples, t_smooth), :x)

#     # Smooth the inner (z) component using backward_smooth
#     inner_dyn = hier_model.inner_model.dyn
#     z_smoothed_means = Vector{T}(undef, N_sample)
#     for i in 1:N_sample
#         smoothed_z = trajectory_samples[i][K].z

#         for t in (K - 1):-1:t_smooth
#             filtered_z = trajectory_samples[i][t].z
#             # Pass prev_outer to condition the inner dynamics on the outer trajectory
#             smoothed_z = backward_smooth(
#                 inner_dyn,
#                 KF(),
#                 t,
#                 filtered_z,
#                 smoothed_z;
#                 prev_outer=trajectory_samples[i][t].x,
#             )
#         end

#         z_smoothed_means[i] = only(smoothed_z.μ)
#     end

#     # Compare to ground truth
#     @test state.μ[1] ≈ only(mean(x_trajectories)) rtol = 1e-2
#     @test state.μ[2] ≈ mean(z_smoothed_means) rtol = 1e-3
# end

# @testitem "Backward simulation test" begin
#     using GeneralisedFilters
#     using StableRNGs
#     using PDMats
#     using LinearAlgebra
#     using StatsBase: sample, weights
#     using Statistics
#     using LogExpFunctions

#     import SSMProblems: dyn, obs, prior
#     import GeneralisedFilters: resample, resampler, move, Particle

#     SEED = 1234
#     Dx = 1
#     Dy = 1
#     K = 5
#     t_smooth = 3
#     T = Float64
#     N_particles = 50
#     N_trajectories = 1000

#     rng = StableRNG(SEED)
#     model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, Dx, Dy)
#     _, _, ys = sample(rng, model, K)

#     # Kalman smoother ground truth
#     ks_state, _ = GeneralisedFilters.smooth(
#         rng, model, KalmanSmoother(), ys; t_smooth=t_smooth
#     )

#     # Run forward filter manually and store particle states at each time step
#     bf = BF(N_particles)

#     # Storage for particle states at time steps 1:K
#     particle_states = Vector{Vector{Particle{Vector{T},T,Int}}}(undef, K)

#     # Forward filtering pass
#     final_state = let state = initialise(rng, prior(model), bf)
#         for t in 1:K
#             state = resample(rng, resampler(bf), state)
#             state, _ = move(rng, model, bf, t, state, ys[t])
#             particle_states[t] = deepcopy(collect(state.particles))
#         end
#         state
#     end

#     # Backward simulation: sample M trajectories
#     trajectory_samples = Vector{Vector{T}}(undef, N_trajectories)

#     for m in 1:N_trajectories
#         # Sample from final distribution
#         final_ws = weights(final_state)
#         idx = sample(rng, 1:N_particles, final_ws)

#         # Initialize trajectory with sampled final state
#         traj = Vector{Vector{T}}(undef, K)
#         traj[K] = particle_states[K][idx].state

#         # Backward simulation pass - resample ancestors using backward weights
#         for t in (K - 1):-1:1
#             particles_t = particle_states[t]

#             # Compute backward weights: w_t^i * f(x_{t+1} | x_t^i)
#             ref_state = traj[t + 1]
#             backward_ws = map(particles_t) do particle
#                 ancestor_weight(particle, dyn(model), bf, t + 1, ref_state)
#             end

#             # Sample new ancestor
#             idx = sample(rng, 1:N_particles, weights(softmax(backward_ws)))
#             traj[t] = particles_t[idx].state
#         end

#         trajectory_samples[m] = [traj[t][1] for t in 1:K]
#     end

#     # Extract samples at t_smooth and compare to Kalman smoother
#     bs_mean = mean(getindex.(trajectory_samples, t_smooth))
#     @test bs_mean ≈ only(ks_state.μ) rtol = 5e-2
# end

# @testitem "RB backward simulation test" begin
#     using GeneralisedFilters
#     using StableRNGs
#     using PDMats
#     using LinearAlgebra
#     using StatsBase: sample, weights
#     using Statistics
#     using LogExpFunctions
#     using Distributions: MvNormal

#     import SSMProblems: dyn, obs, prior
#     import GeneralisedFilters:
#         RBState, InformationLikelihood, resample, resampler, move, Particle

#     SEED = 1234
#     D_outer = 1
#     D_inner = 1
#     D_obs = 1
#     K = 5
#     t_smooth = 2
#     T = Float64
#     N_particles = 50
#     N_trajectories = 1000

#     rng = StableRNG(SEED)
#     full_model, hier_model = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
#         rng, D_outer, D_inner, D_obs, T; static_arrays=false
#     )
#     _, _, ys = sample(rng, full_model, K)

#     # Kalman smoother ground truth on full model
#     ks_state, _ = GeneralisedFilters.smooth(
#         rng, full_model, KalmanSmoother(), ys; t_smooth=t_smooth
#     )

#     # Run RBPF forward filter manually and store particle states
#     rbpf = RBPF(BF(N_particles), KalmanFilter())

#     # Initialize and run first step to get concrete types
#     init_state = initialise(rng, prior(hier_model), rbpf)
#     init_state = resample(rng, resampler(rbpf), init_state)
#     init_state, _ = move(rng, hier_model, rbpf, 1, init_state, ys[1])

#     # Storage for particle states at time steps 1:K
#     particle_states = Vector{typeof(collect(init_state.particles))}(undef, K)
#     particle_states[1] = deepcopy(collect(init_state.particles))

#     # Forward filtering pass for remaining steps
#     final_state = let state = init_state
#         for t in 2:K
#             state = resample(rng, resampler(rbpf), state)
#             state, _ = move(rng, hier_model, rbpf, t, state, ys[t])
#             particle_states[t] = deepcopy(collect(state.particles))
#         end
#         state
#     end

#     # Backward simulation: sample M trajectories
#     x_samples = Vector{T}(undef, N_trajectories)
#     z_samples = Vector{T}(undef, N_trajectories)

#     for m in 1:N_trajectories
#         # Sample from final distribution
#         final_ws = weights(final_state)
#         idx = sample(rng, 1:N_particles, final_ws)

#         # Initialize trajectory with sampled final state
#         traj = Vector{eltype(particle_states[1]).parameters[1]}(undef, K)
#         traj[K] = particle_states[K][idx].state

#         # Extract outer trajectory for computing backward likelihoods
#         outer_traj = Vector{Vector{T}}(undef, K)
#         outer_traj[K] = traj[K].x

#         # Compute backward predictive likelihoods for this trajectory
#         bip = BackwardInformationPredictor(; initial_jitter=1e-8)
#         pred_lik = backward_initialise(rng, hier_model.inner_model.obs, bip, K, ys[K])
#         predictive_likelihoods = Vector{typeof(pred_lik)}(undef, K)
#         predictive_likelihoods[K] = deepcopy(pred_lik)

#         # Backward simulation pass
#         for t in (K - 1):-1:1
#             particles_t = particle_states[t]

#             # Build reference state with backward predictive likelihood
#             ref_rb_state = RBState(outer_traj[t + 1], predictive_likelihoods[t + 1])

#             # Compute backward weights using ancestor_weight
#             backward_ws = map(particles_t) do particle
#                 ancestor_weight(particle, dyn(hier_model), rbpf, t, ref_rb_state)
#             end

#             # Sample new ancestor
#             new_idx = sample(rng, 1:N_particles, weights(softmax(backward_ws)))
#             traj[t] = particles_t[new_idx].state
#             outer_traj[t] = traj[t].x

#             # Compute backward predictive likelihood at time t
#             pred_lik = backward_predict(
#                 rng,
#                 hier_model.inner_model.dyn,
#                 bip,
#                 t,
#                 predictive_likelihoods[t + 1];
#                 prev_outer=outer_traj[t],
#                 new_outer=outer_traj[t + 1],
#             )
#             pred_lik = backward_update(hier_model.inner_model.obs, bip, t, pred_lik, ys[t])
#             predictive_likelihoods[t] = deepcopy(pred_lik)
#         end

#         # Store outer state sample at t_smooth
#         x_samples[m] = only(traj[t_smooth].x)

#         # Smooth the inner (z) component using backward_smooth
#         inner_dyn = hier_model.inner_model.dyn
#         smoothed_z = traj[K].z
#         for t in (K - 1):-1:t_smooth
#             filtered_z = traj[t].z
#             smoothed_z = backward_smooth(
#                 inner_dyn, KF(), t, filtered_z, smoothed_z; prev_outer=traj[t].x
#             )
#         end
#         z_samples[m] = only(smoothed_z.μ)
#     end

#     # Compare to ground truth
#     @test ks_state.μ[1] ≈ mean(x_samples) rtol = 5e-2
#     @test ks_state.μ[2] ≈ mean(z_samples) rtol = 5e-2
# end
