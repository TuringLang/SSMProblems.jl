@testitem "AD: Kalman covariance storage and asymmetric output seed" tags = [:mooncake] begin
    using GeneralisedFilters
    using GeneralisedFilters.GFTest: central_diff, check_gradients
    using StaticArrays
    using ForwardDiff
    using Mooncake

    # Each covariance entry is an independent parameter. Although the starting
    # covariances are symmetric, a storage derivative varies each triangle alone.
    # Reading Σ[1,2] supplies an asymmetric output cotangent to the fused step.
    function objective(θ)
        Σ0 = SMatrix{2,2}(θ[1:4])
        Q = SMatrix{2,2}(θ[5:8])
        R = SMatrix{2,2}(θ[9:12])
        state = GaussianState(SA[0.1, 0.2], Σ0)
        dynamics = LinearGaussianDynamics(SA[0.8 0.1; -0.1 0.9], SA[0.1, -0.2], Q)
        observation = LinearGaussianObservation(SA[0.9 0.2; -0.3 1.1], SA[0.0, 0.1], R)
        filtered, ll = GeneralisedFilters.kalman_step(
            state, dynamics, observation, SA[0.4, 0.5]
        )
        return filtered.Σ[1, 2] + 0.3 * filtered.μ[1] + 0.2 * ll
    end
    θ = [1.0, 0.2, 0.2, 0.8, 0.3, 0.03, 0.03, 0.2, 0.4, 0.01, 0.01, 0.5]
    forward = ForwardDiff.gradient(objective, θ)
    reverse = check_gradients(objective, θ)
    @test reverse.agrees
    @test reverse.ad ≈ forward rtol = 1e-8 atol = 1e-10
    @test forward ≈ central_diff(objective, θ) rtol = 1e-6 atol = 1e-9
    for offset in (0, 4, 8)
        @test reverse.ad[offset + 2] ≈ reverse.ad[offset + 3] atol = 1e-12
    end
end
