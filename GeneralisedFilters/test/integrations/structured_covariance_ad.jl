@testitem "AD: structured covariance parameters compose and accumulate" tags = [:mooncake] begin
    using GeneralisedFilters
    using GeneralisedFilters.GFTest: central_diff
    using LinearAlgebra
    using StaticArrays
    using ForwardDiff
    using Mooncake

    representation(θ, ::Val{:upper}) = Symmetric(reshape(θ, 2, 2), :U)
    representation(θ, ::Val{:lower}) = Symmetric(reshape(θ, 2, 2), :L)
    representation(θ, ::Val{:hermitian_upper}) = Hermitian(reshape(θ, 2, 2), :U)
    representation(θ, ::Val{:hermitian_lower}) = Hermitian(reshape(θ, 2, 2), :L)
    representation(θ, ::Val{:diagonal}) = Diagonal(θ)
    representation(θ, ::Val{:static_diagonal}) = Diagonal(SVector{2}(θ))
    representation(θ, ::Val{:static_upper}) = Symmetric(SMatrix{2,2}(θ), :U)

    storage_array(::Val{true}, x) = x
    storage_array(::Val{false}, x) = Array(x)

    function objective(θ, rep, storage)
        arr(x) = storage_array(storage, x)
        covariance = representation(θ, rep)
        # Exactly the same covariance object appears in all three slots.
        # Mutable parent gradients must accumulate, including across time.
        model = StateSpaceModel(
            GaussianPrior(arr(SA[0.1, 0.2]), covariance),
            LinearGaussianDynamics(
                arr(SA[0.8 0.2; 0.0 0.8]), arr(SA[0.0, 0.0]), covariance
            ),
            LinearGaussianObservation(
                arr(SA[1.0 0.1; 0.0 1.0]), arr(SA[0.0, 0.0]), covariance
            ),
        )
        return marginal_loglikelihood(model, KF(), [arr(SA[0.4, -0.7]), arr(SA[0.2, 0.8])])
    end

    for (kind, θ, unused, static) in (
        (:upper, [1.7, -0.6, 0.3, 1.2], 2, true),
        (:upper, [1.7, -0.6, 0.3, 1.2], 2, false),
        (:lower, [1.7, 0.3, -0.6, 1.2], 3, false),
        (:hermitian_upper, [1.7, -0.6, 0.3, 1.2], 2, false),
        (:hermitian_lower, [1.7, 0.3, -0.6, 1.2], 3, true),
        (:diagonal, [1.7, 1.2], 0, false),
        (:static_diagonal, [1.7, 1.2], 0, true),
        (:static_upper, [1.7, -0.6, 0.3, 1.2], 2, true),
    )
        rep, storage = Val(kind), Val(static)
        f(x) = objective(x, rep, storage)
        forward = ForwardDiff.gradient(f, θ)
        @test forward ≈ central_diff(f, θ) rtol = 1e-6 atol = 1e-8
        for composed in (f, x -> f(x) + sum(x), x -> f(x) + f(x))
            cache = Mooncake.prepare_gradient_cache(composed, θ)
            for x in (θ, θ .+ 0.02)
                original = copy(x)
                value, (_, reverse) = Mooncake.value_and_gradient!!(cache, composed, x)
                @test value ≈ composed(x)
                @test reverse ≈ ForwardDiff.gradient(composed, x) rtol = 1e-8 atol = 1e-10
                @test x == original
            end
        end
        if unused != 0
            cache = Mooncake.prepare_gradient_cache(f, θ)
            _, (_, reverse) = Mooncake.value_and_gradient!!(cache, f, θ)
            @test reverse[unused] == 0
        end
    end
end

@testitem "AD: mixed-precision Kalman likelihood promotion" tags = [:mooncake] begin
    using GeneralisedFilters
    using StaticArrays
    using ForwardDiff
    using Mooncake

    function objective(θ)
        model = StateSpaceModel(
            GaussianPrior(SVector{2}(θ), SA[1.0f0 0.0f0; 0.0f0 1.0f0]),
            LinearGaussianDynamics(
                SA[0.8 0.0; 0.0 0.8], SA[0.0f0, 0.0f0], SA[0.5 0.0; 0.0 0.5]
            ),
            LinearGaussianObservation(
                SA[1.0f0 0.0f0; 0.0f0 1.0f0], SA[0.0f0, 0.0f0], SA[0.3 0.0; 0.0 0.3]
            ),
        )
        return marginal_loglikelihood(model, KF(), [SA[0.4f0, -0.7f0], SA[0.2f0, 0.8f0]])
    end
    θ = Float32[0.1, 0.2]
    cache = Mooncake.prepare_gradient_cache(objective, θ)
    value, (_, reverse) = Mooncake.value_and_gradient!!(cache, objective, θ)
    @test value ≈ objective(θ)
    @test eltype(reverse) === Float32
    @test reverse ≈ ForwardDiff.gradient(objective, θ) rtol = 2e-6 atol = 1e-7
end

@testitem "AD: covariance-only promotion and empty Kalman likelihood" tags = [:mooncake] begin
    using GeneralisedFilters
    using StaticArrays
    using LinearAlgebra
    using ForwardDiff
    using Mooncake

    function objective(θ, T)
        # Mean constants must acquire the covariance's Dual/Float64 scalar type
        # during state construction, without changing the original prior.
        model = StateSpaceModel(
            GaussianPrior(SA[0.0f0, 0.0f0], Diagonal(exp.(SVector{2}(θ)))),
            LinearGaussianDynamics(
                SA[0.8f0 0.1f0; 0.0f0 0.8f0], SA[0.0f0, 0.0f0], Diagonal(SA[0.5f0, 0.4f0])
            ),
            LinearGaussianObservation(
                SA[1.0f0 0.0f0; 0.0f0 1.0f0], SA[0.0f0, 0.0f0], Diagonal(SA[0.3f0, 0.2f0])
            ),
        )
        ys = fill(SA[0.4f0, -0.7f0], T)
        return marginal_loglikelihood(
            model, KalmanFilter(; repair=Jitter(0.001exp(θ[1]))), ys
        )
    end
    for θ in (Float32[0.1, 0.2], [0.1, 0.2]), T in (0, 1, 3)
        f(x) = objective(x, T)
        cache = Mooncake.prepare_gradient_cache(f, θ)
        value, (_, reverse) = Mooncake.value_and_gradient!!(cache, f, θ)
        @test value isa Float64
        @test value ≈ f(θ)
        @test reverse ≈ ForwardDiff.gradient(f, θ) rtol = 2e-6 atol = 1e-7
        if T == 0
            @test value == 0
            @test iszero(reverse)
        end
    end
end

@testitem "AD: views and immutable Kalman input containers" tags = [:mooncake] begin
    using GeneralisedFilters
    using StaticArrays
    using ForwardDiff
    using Mooncake

    function objective(θ, container)
        # Share a view between the prior mean and both offsets. Its parent must
        # receive all contributions; unused parent entries must stay inactive.
        μ = view(θ, 2:3)
        A = SA[0.8 0.1; 0.0 0.8]
        Q = SA[0.5 0.0; 0.0 0.4]
        R = SA[0.3 0.0; 0.0 0.2]
        dyns = SVector(LinearGaussianDynamics(A, μ, Q), LinearGaussianDynamics(A, μ, Q))
        observations = [view(θ, 4:5), view(θ, 6:7)]
        ys = container(observations)
        model = StateSpaceModel(
            GaussianPrior(μ, Q),
            TimeVaryingDynamics(ctx -> dyns[ctx.t]),
            LinearGaussianObservation(SA[1.0 0.0; 0.0 1.0], μ, R),
        )
        return marginal_loglikelihood(model, KF(), ys)
    end

    for container in (identity, ys -> view(ys, :), ys -> SVector{2}(ys))
        f(x) = objective(x, container)
        θ = [99.0, 0.1, 0.2, 0.4, -0.7, 0.2, 0.8, 99.0]
        for composed in (f, x -> f(x) + sum(x), x -> f(x) + f(x))
            cache = Mooncake.prepare_gradient_cache(composed, θ)
            for x in (θ, θ .+ 0.01)
                original = copy(x)
                value, (_, reverse) = Mooncake.value_and_gradient!!(cache, composed, x)
                @test value ≈ composed(x)
                @test reverse ≈ ForwardDiff.gradient(composed, x) rtol = 1e-8 atol = 1e-10
                @test x == original
            end
        end
        @test ForwardDiff.gradient(f, θ)[[1, 8]] == [0.0, 0.0]
    end
end
