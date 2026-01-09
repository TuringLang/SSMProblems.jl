import LinearAlgebra: I, cholesky, logdet, dot
using PDMats: PDMats

export future_conditional_density, ancestor_weight

@doc raw"""
    future_conditional_density(dyn, algo, iter, state, ref_state; kwargs...)

Compute the log conditional density of the future trajectory given the present state:
```math
\log p(x_{t+1:T}, y_{t+1:T} \mid x_{1:t}, y_{1:t})
```
up to an additive constant that does not depend on ``x_t``.

This function is the key computational primitive for backward simulation (BS) and ancestor
sampling (AS) algorithms. The full backward sampling weight combines this with the filtering
weight:
```math
\tilde{w}_{t|T}^{(i)} \propto w_t^{(i)} \cdot p(x_{t+1:T}^*, y_{t+1:T} \mid x_{1:t}^{(i)}, y_{1:t})
```

# Standard (Markov) Case

For Markovian models, the future conditional density factorizes as:
```math
p(x_{t+1:T}, y_{t+1:T} \mid x_{1:t}, y_{1:t}) = f(x_{t+1} \mid x_t) \cdot p(y_{t+1:T} \mid x_{t+1:T})
```
The second factor is constant across candidate ancestors (since ``x_{t+1:T}^*`` is fixed),
so this function returns only the transition density:
```math
\texttt{future\_conditional\_density} = \log f(x_{t+1}^* \mid x_t)
```

# Rao-Blackwellised Case

For hierarchical models with Rao-Blackwellisation, the marginal outer state process is
non-Markov due to the marginalized inner state. The future conditional density becomes:
```math
p(u_{t+1:T}, y_{t+1:T} \mid u_{1:t}, y_{1:t}) \propto f(u_{t+1} \mid u_t) \cdot p(y_{t+1:T} \mid u_{1:T}, y_{1:t})
```
where ``u`` denotes the outer (sampled) state. Unlike the Markov case, the second factor
depends on the candidate ancestor through ``u_{1:t}``. This is computed via the two-filter
formula using the forward filtering distribution and backward predictive likelihood.

# Arguments
- `dyn`: The latent dynamics model
- `algo`: The filtering algorithm
- `iter::Integer`: The time step t
- `state`: The candidate state at time t (contains filtering distribution for RB case)
- `ref_state`: The reference trajectory state at time t+1

# Returns
The log future conditional density (up to additive constants independent of ``x_t``).

# Implementations
- **Generic** (`LatentDynamics`, `AbstractFilter`): Returns `logdensity(dyn, iter, state, ref_state)`
- **Rao-Blackwellised** (`HierarchicalDynamics`, `RBPF`): Combines outer transition density with
  marginal predictive likelihood. The `ref_state.z` must be an `AbstractLikelihood`
  (`InformationLikelihood` for Gaussian inner states, `DiscreteLikelihood` for discrete inner states).

See also: [`compute_marginal_predictive_likelihood`](@ref), [`BackwardInformationPredictor`](@ref),
[`BackwardDiscretePredictor`](@ref)
"""
function future_conditional_density(
    dyn::LatentDynamics, ::AbstractFilter, iter::Integer, state, ref_state; kwargs...
)
    return SSMProblems.logdensity(dyn, iter, state, ref_state; kwargs...)
end

function future_conditional_density(
    dyn::HierarchicalDynamics,
    algo::RBPF,
    iter::Integer,
    state::RBState,
    ref_state::RBState{<:Any,<:AbstractLikelihood};
    kwargs...,
)
    trans_density = future_conditional_density(
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
    return trans_density + marginal_pred_lik
end

@doc raw"""
    ancestor_weight(particle, dyn, algo, iter, ref_state; kwargs...)

Compute the full (unnormalized) log backward sampling weight for a particle.

This is a convenience function that combines the particle's filtering log-weight with the
future conditional density:
```math
\log \tilde{w}_{t|T}^{(i)} = \log w_t^{(i)} + \log p(x_{t+1:T}^*, y_{t+1:T} \mid x_{1:t}^{(i)}, y_{1:t})
```

# Arguments
- `particle::Particle`: The candidate particle at time t
- `dyn`: The latent dynamics model
- `algo`: The filtering algorithm
- `iter::Integer`: The time step t
- `ref_state`: The reference trajectory state at time t+1

# Returns
The log backward sampling weight (unnormalized).

See also: [`future_conditional_density`](@ref)
"""
function ancestor_weight(particle::Particle, dyn, algo, iter::Integer, ref_state; kwargs...)
    return log_weight(particle) +
           future_conditional_density(dyn, algo, iter, particle.state, ref_state; kwargs...)
end

"""
    compute_marginal_predictive_likelihood(forward_dist, backward_dist)

Compute the marginal predictive likelihood p(y_{t:T} | y_{1:t-1}) given a one-step predicted
filtering distribution p(x_{t+1} | y_{1:t}) and a backward predictive likelihood
p(y_{t+1:T} | x_{t+1}).

This Gaussian implementation is based on Lemma 1 of https://arxiv.org/pdf/1505.06357
"""
function compute_marginal_predictive_likelihood(
    forward_dist::MvNormal, backward_dist::InformationLikelihood
)
    μ, Σ = params(forward_dist)
    λ, Ω = natural_params(backward_dist)
    Γ = cholesky(Σ).L

    # Apply two-filter smoother style formula
    Λ = PDMat(Xt_A_X(Ω, Γ).data + I)
    M = Γ' * (λ - Ω * μ)
    ζ = PDMats.quad(Ω, μ) - 2 * dot(λ, μ) - PDMats.invquad(Λ, M)

    return -0.5 * (logdet(Λ) + ζ)
end

"""
    compute_marginal_predictive_likelihood(forward_dist::AbstractVector, backward_dist::DiscreteLikelihood)

Compute the marginal predictive likelihood p(y_{t+1:T} | y_{1:t}) for discrete states.

Given a predicted filtering distribution π_{t+1}(i) = p(x_{t+1} = i | y_{1:t}) and backward
likelihood β_{t+1}(i) = p(y_{t+1:T} | x_{t+1} = i), computes:

    p(y_{t+1:T} | y_{1:t}) = Σ_i π_{t+1}(i) * β_{t+1}(i)

All computations are performed in log-space for numerical stability.
"""
function compute_marginal_predictive_likelihood(
    forward_dist::AbstractVector, backward_dist::DiscreteLikelihood
)
    log_forward = log.(forward_dist)
    log_β = log_likelihoods(backward_dist)
    return logsumexp(log_forward .+ log_β)
end
