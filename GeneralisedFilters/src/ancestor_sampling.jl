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

    # Apply two filter formula to get p(y_{t+1:T} | y_{1:t}, x_{1:t})
    # Or is it p(y_{t+1:T} | y_{1:t}, x_{1:T})?
    μ, Σ = mean_cov(pred_dist)
    λ, Ω = natural_params(back_info)
    Γ = cholesky(Σ).L

    # Apply two-filter smoother style formula
    # TODO: Clean this up with Mahalanobis distance helper
    Λ = Γ' * Ω * Γ + I
    M = Γ' * (λ - Ω * μ)
    ζ = μ' * Ω * μ - 2 * λ' * μ - M' * inv(Λ) * M

    return trans_weight + -0.5 * (logdet(Λ) + ζ)
end
