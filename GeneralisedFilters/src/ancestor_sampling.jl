import LinearAlgebra: I, cholesky, logdet

export ancestor_weight

function ancestor_weight(
    dyn::LatentDynamics, ::AbstractFilter, iter::Integer, state, ref_state; kwargs...
)
    return SSMProblems.logdensity(dyn, iter, state, ref_state; kwargs...)
end

function ancestor_weight(
    dyn::HierarchicalDynamics,
    algo::RBPF,
    iter::Integer,
    state::RBState,
    ref_state::RBState{<:Any,<:InformationDistribution};
    kwargs...,
)
    trans_weight = ancestor_weight(
        dyn.outer_dyn, algo.pf, iter, state.x, ref_state.x; kwargs...
    )
    filt_dist = state.z
    # A representation of the predictive likelihood p(y_{t+1:T} | z_{t+1}) conditioned on
    # the reference trajectory
    back_info = ref_state.z

    # Predict filtering distribution using reference state
    # TODO: this is wasteful if prediction doesn't depend on the new outer state
    pred_dist = predict(
        default_rng(),
        dyn.inner_dyn,
        algo.af,
        iter,
        filt_dist,
        nothing;
        prev_outer=state.x,
        new_outer=ref_state.x,
        kwargs...,
    )

    marginal_pred_lik = compute_marginal_predictive_likelihood(pred_dist, back_info)
    return trans_weight + marginal_pred_lik
end

"""
    compute_marginal_predictive_likelihood(forward_dist, backward_dist)

Compute the marginal predictive likelihood p(y_{t:T} | y_{1:t-1}) given a one-step predicted
filtering distribution p(x_{t+1} | y_{1:t}) and a backward predictive likelihood
p(y_{t+1:T} | x_{t+1}).

This Gaussian implementation is based on Lemma 1 of https://arxiv.org/pdf/1505.06357
"""
function compute_marginal_predictive_likelihood(
    forward_dist::GaussianDistribution, backward_dist::InformationDistribution
)
    μ, Σ = mean_cov(forward_dist)
    λ, Ω = natural_params(backward_dist)
    Γ = cholesky(Σ).L

    # Apply two-filter smoother style formula
    # TODO: Clean this up with Mahalanobis distance helper
    Λ = Γ' * Ω * Γ + I
    M = Γ' * (λ - Ω * μ)
    ζ = μ' * Ω * μ - 2 * λ' * μ - M' * inv(Λ) * M

    return -0.5 * (logdet(Λ) + ζ)
end
