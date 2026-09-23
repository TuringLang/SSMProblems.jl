"""Gradients of the complete fixed-trajectory parameter objective."""

@testsnippet ConditionalADSetup begin
    using GeneralisedFilters
    using GeneralisedFilters.GFTest: central_diff
    using Distributions
    using StaticArrays
    using ForwardDiff

    # Every Gaussian field varies with θ. Shared parameters also enter the outer law,
    # so an inner-only gradient cannot pass these checks. All covariance constructions
    # remain positive definite under the finite-difference perturbations.
    function conditional_ad_model(θ)
        function inner_prior((; x0))
            μ = SA[θ[1] + x0, θ[2] * x0]
            L = SA[exp(θ[3]) 0.0; θ[4] exp(θ[5])]
            return GaussianPrior(μ, L * L')
        end
        function inner_dynamics((; t, x_prev, x_new))
            A = SA[0.65+0.1tanh(θ[6]) θ[7]*x_new; 0.03x_prev 0.7]
            b = SA[θ[2] * x_prev + 0.01t, θ[8] * x_new]
            L = SA[exp(θ[4]) 0.0; 0.1θ[6] exp(θ[9])]
            return LinearGaussianDynamics(A, b, L * L')
        end
        function inner_observation((; t, x))
            H = SA[1.0 + θ[7] * x θ[8] + 0.01t]
            c = SA[θ[1] * x + θ[9]]
            R = SMatrix{1,1}(exp(θ[10]))
            return LinearGaussianObservation(H, c, R)
        end
        return StateSpaceModel(
            HierarchicalPrior(DistributionPrior(Normal(θ[1], exp(θ[3]))), inner_prior),
            HierarchicalDynamics(
                DistributionDynamics(
                    (t, x) -> Normal(tanh(θ[6]) * x + 0.02t * θ[2], exp(θ[5]))
                ),
                inner_dynamics,
            ),
            HierarchicalObservation(inner_observation),
        )
    end

    θ0 = [0.13, -0.21, -0.35, 0.08, -0.42, 0.27, -0.17, 0.11, -0.51, -0.63]
    xs = [0.2, -0.15, 0.35, 0.1, -0.25, 0.4]
    ys = [SA[0.3], SA[-0.1], SA[0.7], SA[-0.4], SA[0.2]]
    ref = ReferenceTrajectory(first(xs), xs[2:end])
    changed = ReferenceTrajectory(-0.3, [0.45, -0.2, 0.3, 0.5, -0.1])

    # A simple parameter prior represents the contribution owned by the caller/Turing.
    parameter_objective(θ, x) =
        trajectory_logdensity(conditional_ad_model(θ), KalmanFilter(), x, ys) -
        sum(abs2, θ) / 2
    objective(θ) = parameter_objective(θ, xs)
    reference_objective(θ) = parameter_objective(θ, ref)
    changed_objective(θ) = parameter_objective(θ, changed)
end

@testitem "Conditional objective: StaticArrays ForwardDiff and refreshed trajectory" setup = [
    ConditionalADSetup
] begin
    gradient = ForwardDiff.gradient(objective, θ0)
    reference_gradient = ForwardDiff.gradient(reference_objective, θ0)
    changed_gradient = ForwardDiff.gradient(changed_objective, θ0)

    @test objective(θ0) ≈ reference_objective(θ0)
    @test gradient ≈ central_diff(objective, θ0) rtol = 1e-6 atol = 1e-8
    @test reference_gradient ≈ gradient rtol = 1e-12 atol = 1e-12
    @test changed_gradient ≈ central_diff(changed_objective, θ0) rtol = 1e-6 atol = 1e-8
    @test !isapprox(changed_objective(θ0), objective(θ0))
    @test !isapprox(changed_gradient, gradient)

    # The independently evaluated outer contribution must reach the parameter gradient.
    inner_objective(θ) =
        inner_loglikelihood(KalmanFilter(), conditional_ad_model(θ), xs, ys)
    outer_objective(θ) = outer_logdensity(conditional_ad_model(θ), xs)
    inner_gradient = ForwardDiff.gradient(inner_objective, θ0)
    outer_gradient = ForwardDiff.gradient(outer_objective, θ0)
    @test gradient ≈ inner_gradient + outer_gradient - θ0
    @test maximum(abs, outer_gradient) > 0.1
end

@testitem "Conditional objective: Mooncake agrees with ForwardDiff and finite differences" setup = [
    ConditionalADSetup
] tags = [:mooncake] begin
    using Mooncake
    using GeneralisedFilters.GFTest: check_gradients

    for f in (objective, reference_objective, changed_objective)
        result = check_gradients(f, θ0)
        @test result.agrees
        @test result.ad ≈ ForwardDiff.gradient(f, θ0) rtol = 1e-7 atol = 1e-8
    end
end

@testitem "Custom hierarchical model: forward and reverse parameter gradients" setup = [
    ConditionalADSetup
] tags = [:mooncake] begin
    using SSMProblems, Mooncake
    using GeneralisedFilters.GFTest: check_gradients
    struct CustomConditionalModel{T} <: AbstractStateSpaceModel
        components::T
    end
    SSMProblems.prior(m::CustomConditionalModel) = m.components[1]
    SSMProblems.dyn(m::CustomConditionalModel) = m.components[2]
    SSMProblems.obs(m::CustomConditionalModel) = m.components[3]
    function custom_objective(θ)
        model = conditional_ad_model(θ)
        custom = CustomConditionalModel((prior(model), dyn(model), obs(model)))
        return trajectory_logdensity(custom, KF(), xs, ys) - sum(abs2, θ) / 2
    end
    @test custom_objective(θ0) ≈ objective(θ0)
    @test ForwardDiff.gradient(custom_objective, θ0) ≈ ForwardDiff.gradient(objective, θ0)
    result = check_gradients(custom_objective, θ0)
    @test result.agrees
    @test result.ad ≈ ForwardDiff.gradient(objective, θ0) rtol = 1e-7 atol = 1e-8
end
