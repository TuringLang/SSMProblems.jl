using TestItems
using TestItemRunner

@run_package_tests

@testitem "Kalman filter test" begin
    using AnalyticFilters
    using LinearAlgebra
    using StableRNGs

    rng = StableRNG(1234)
    μ0 = rand(rng, 2)
    Σ0 = rand(rng, 2, 2)
    Σ0 = Σ0 * Σ0'  # make Σ0 positive definite
    A = rand(rng, 2, 2)
    b = rand(rng, 2)
    Q = rand(rng, 2, 2)
    Q = Q * Q'  # make Q positive definite
    H = rand(rng, 2, 2)
    c = rand(rng, 2)
    R = rand(rng, 2, 2)
    R = R * R'  # make R positive definite

    model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)

    observations = [rand(rng, 2)]

    kf = KalmanFilter()

    states, _ = AnalyticFilters.filter(model, kf, observations, [nothing])

    # Let Z = [X0, X1, Y1] be the joint state vector
    # Write Z = P.Z + ϵ, where ϵ ~ N(μ_ϵ, Σ_ϵ)
    P = [
        zeros(2, 6)
        A zeros(2, 4)
        zeros(2, 2) H zeros(2, 2)
    ]
    μ_ϵ = [μ0; b; c]
    Σ_ϵ = [
        Σ0 zeros(2, 4)
        zeros(2, 2) Q zeros(2, 2)
        zeros(2, 4) R
    ]

    # Note (I - P)Z = ϵ and solve for Z ~ N(μ_Z, Σ_Z)
    μ_Z = (I - P) \ μ_ϵ
    Σ_Z = ((I - P) \ Σ_ϵ) / (I - P)'

    # Condition on observations using formula for MVN conditional distribution. See: 
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    y = only(observations)
    I_x = 3:4
    I_y = 5:6
    μ_X1 = μ_Z[I_x] + Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ (y - μ_Z[I_y]))
    Σ_X1 = Σ_Z[I_x, I_x] - Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ Σ_Z[I_y, I_x])

    # TODO: test log-likelihood using marginalisation formula
    @test only(states).μ ≈ μ_X1
    @test only(states).Σ ≈ Σ_X1
end

@testitem "Kalman-RBPF test" begin
    using AnalyticFilters
    using Distributions
    using HypothesisTests
    using LinearAlgebra
    using LogExpFunctions: softmax
    using StableRNGs
    using StatsBase

    # Define inner dynamics
    struct InnerDynamics{T} <: LinearGaussianLatentDynamics{T}
        μ0::Vector{T}
        Σ0::Matrix{T}
        A::Matrix{T}
        b::Vector{T}
        C::Matrix{T}
        Q::Matrix{T}
    end
    AnalyticFilters.calc_μ0(dyn::InnerDynamics) = dyn.μ0
    AnalyticFilters.calc_Σ0(dyn::InnerDynamics) = dyn.Σ0
    AnalyticFilters.calc_A(dyn::InnerDynamics, ::Integer, extra) = dyn.A
    function AnalyticFilters.calc_b(dyn::InnerDynamics, ::Integer, extra)
        return dyn.b + dyn.C * extra.prev_outer
    end
    AnalyticFilters.calc_Q(dyn::InnerDynamics, ::Integer, extra) = dyn.Q

    rng = StableRNG(1234)
    μ0 = rand(rng, 4)
    Σ0s = [rand(rng, 2, 2) for _ in 1:2]
    Σ0s = [Σ * Σ' for Σ in Σ0s]  # make Σ0 positive definite
    Σ0 = [
        Σ0s[1] zeros(2, 2)
        zeros(2, 2) Σ0s[2]
    ]
    A = [
        rand(rng, 2, 2) zeros(2, 2)
        rand(rng, 2, 4)
    ]
    # Make mean-reverting
    A /= 3.0
    A[diagind(A)] .= -0.5
    b = rand(rng, 4)
    Qs = [rand(rng, 2, 2) / 10.0 for _ in 1:2]
    Qs = [Q * Q' for Q in Qs]  # make Q positive definite
    Q = [
        Qs[1] zeros(2, 2)
        zeros(2, 2) Qs[2]
    ]
    H = [zeros(2, 2) rand(rng, 2, 2)]
    c = rand(rng, 2)
    R = rand(rng, 2, 2)
    R = R * R' / 3.0  # make R positive definite

    N_particles = 100000
    T = 1

    observations = [rand(rng, 2) for _ in 1:T]
    extras = [nothing for _ in 1:T]

    # Kalman filtering

    full_model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
    kf_states, ll = AnalyticFilters.filter(full_model, KalmanFilter(), observations, extras)

    # Rao-Blackwellised particle filtering

    outer_dyn = AnalyticFilters.HomogeneousLinearGaussianLatentDynamics(
        μ0[1:2], Σ0[1:2, 1:2], A[1:2, 1:2], b[1:2], Qs[1]
    )
    inner_dyn = InnerDynamics(
        μ0[3:4], Σ0[3:4, 3:4], A[3:4, 3:4], b[3:4], A[3:4, 1:2], Qs[2]
    )
    obs = AnalyticFilters.HomogeneousLinearGaussianObservationProcess(H[:, 3:4], c, R)
    hier_model = HierarchicalSSM(outer_dyn, inner_dyn, obs)

    rbpf = RBPF(KalmanFilter(), N_particles)
    (xs, zs, log_ws), ll = AnalyticFilters.filter(
        rng, hier_model, rbpf, observations, extras
    )

    weights = Weights(softmax(log_ws))

    println("ESS: ", 1 / sum(weights .^ 2))
    println("Weighted mean:", sum(xs .* weights))
    println("Kalman filter mean:", kf_states[T].μ[1:2])

    # Resample outer states
    # resampled_xs = sample(rng, xs, weights, N_particles)
    # println(mean(first.(resampled_xs)))
    # test = ExactOneSampleKSTest(
    #     first.(resampled_xs), Normal(kf_states[T].μ[1], sqrt(kf_states[T].Σ[1, 1]))
    # )
    # @test pvalue(test) > 0.05

    println("Weighted mean:", sum(getproperty.(zs, :μ) .* weights))
    println("Kalman filter mean:", kf_states[T].μ[3:4])

    # Resample inner states and demarginalise
    # resampled_zs = sample(rng, zs, weights, N_particles)
    # resampled_inner = [rand(rng, Normal(p.μ[1], sqrt(p.Σ[1, 1]))) for p in resampled_zs]
    # test = ExactOneSampleKSTest(
    #     resampled_inner, Normal(kf_states[T].μ[3], sqrt(kf_states[T].Σ[3, 3]))
    # )

    # @test pvalue(test) > 0.05
end
