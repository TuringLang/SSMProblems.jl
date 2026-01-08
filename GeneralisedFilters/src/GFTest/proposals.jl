using Distributions

"""
    OptimalProposal

Optimal importance proposal for linear Gaussian state space models.

A proposal coming from the closed-form distribution `p(x_t | x_{t-1}, y_t)`. This proposal
minimizes the variance of the importance weights.
"""
struct OptimalProposal{
    LD<:LinearGaussianLatentDynamics,OP<:LinearGaussianObservationProcess
} <: AbstractProposal
    dyn::LD
    obs::OP
end

function SSMProblems.distribution(prop::OptimalProposal, step::Integer, x, y; kwargs...)
    # Get parameters
    dyn_params = calc_params(prop.dyn, step; kwargs...)
    obs_params = calc_params(prop.obs, step; kwargs...)
    A, b, Q = dyn_params

    # Predicted state: p(x_t | x_{t-1})
    state = MvNormal(A * x + b, Q)

    # Update with observation: p(x_t | x_{t-1}, y_t)
    state, _ = GeneralisedFilters.kalman_update(state, obs_params, y, nothing) # no jitter

    return state
end

"""
    OverdispersedProposal

A proposal that overdisperses the latent dynamics by inflating the covariance.
"""
struct OverdispersedProposal{LD<:LinearGaussianLatentDynamics} <: AbstractProposal
    dyn::LD
    k::Float64
end

function SSMProblems.distribution(
    prop::OverdispersedProposal, step::Integer, x, y; kwargs...
)
    A, b, Q = calc_params(prop.dyn, step; kwargs...)
    Q = prop.k * Q  # overdisperse
    μ = A * x + b
    return MvNormal(μ, Q)
end
