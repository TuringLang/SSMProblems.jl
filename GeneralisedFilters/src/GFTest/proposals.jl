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
    dyn_p = GeneralisedFilters.step_eval(prop.dyn, step; kwargs...)
    obs_p = GeneralisedFilters.step_eval(prop.obs, step; kwargs...)

    # Predicted state: p(x_t | x_{t-1})
    state = MvNormal(dyn_p.A * x + dyn_p.b, dyn_p.Q)

    # Update with observation: p(x_t | x_{t-1}, y_t)
    state, _ = GeneralisedFilters.kalman_update(
        state, (obs_p.H, obs_p.c, obs_p.R), y, nothing
    )

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
    p = GeneralisedFilters.step_eval(prop.dyn, step; kwargs...)
    return MvNormal(p.A * x + p.b, prop.k * p.Q)
end
