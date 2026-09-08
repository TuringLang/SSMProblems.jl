export inner_loglikelihood, outer_logdensity, trajectory_logdensity

"""
    inner_loglikelihood(af::AbstractFilter, model::HierarchicalSSM, xs, ys)

Evaluate `log p(y₁:T | x₀:T)` by conditioning the inner model and calling its ordinary
`marginal_loglikelihood` evaluator. No parameter prior or outer trajectory density is added.
The filter must support marginal likelihood evaluation for the resolved inner components.
See [`condition_inner`](@ref) for trajectory indexing and lifetime requirements.
"""
function inner_loglikelihood(
    af::AbstractFilter, model::HierarchicalSSM, xs::AbstractVector, ys::AbstractVector
)
    return marginal_loglikelihood(condition_inner(model, xs), af, ys)
end

"""
    outer_logdensity(model::HierarchicalSSM, xs::AbstractVector)

Compute the outer trajectory log-density `log p(x₀) + Σₜ log p(xₜ | xₜ₋₁)`.
A `ReferenceTrajectory` uses indices `0:T`; a one-based vector stores `x₀` at index 1.
Includes all transitions represented by `xs`, and excludes observations and parameter priors.
"""
function outer_logdensity(model::HierarchicalSSM, xs::AbstractVector)
    _validate_trajectory(xs)
    ll = logdensity(model.prior.outer, _trajectory_state(xs, 0))
    for t in 1:(length(xs) - 1)
        ll += logdensity(
            model.dyn.outer, t, _trajectory_state(xs, t - 1), _trajectory_state(xs, t)
        )
    end
    return ll
end

"""
    trajectory_logdensity(model::HierarchicalSSM, af::AbstractFilter, xs, ys)

Compute `log p(x₀:T, y₁:T)` with the inner states integrated out. This is the sum of
`outer_logdensity` and `inner_loglikelihood`, using the same model definition. Parameters
are supplied by constructing `model`; their prior density and any parameter transformation
Jacobian are deliberately left to the caller (for example, Turing).

Differentiating `θ -> trajectory_logdensity(build(θ), af, xs, ys)` keeps the outer trajectory
fixed while retaining parameter dependence in both the outer and inner processes. This
function does not install a particle-Gibbs sampler or manage Turing/AD caches.
"""
function trajectory_logdensity(
    model::HierarchicalSSM, af::AbstractFilter, xs::AbstractVector, ys::AbstractVector
)
    # Validate the horizon before evaluating outer densities.
    inner_ll = inner_loglikelihood(af, model, xs, ys)
    return outer_logdensity(model, xs) + inner_ll
end

"""
    trajectory_logdensity(model::StateSpaceModel, xs, ys)

Compute the joint log-density of sampled states and observations, including the initial
state and every transition. Unlike the four-argument hierarchical method, this method
integrates out no states: a hierarchical model therefore takes `HierarchicalState` values.
Trajectory indexing follows [`condition_inner`](@ref); observations are one-based.
"""
function trajectory_logdensity(
    model::StateSpaceModel, xs::AbstractVector, ys::AbstractVector
)
    _validate_trajectory(xs)
    _validate_observations(model, ys)
    length(xs) == length(ys) + 1 || throw(
        DimensionMismatch(
            "trajectory must contain one initial state plus one state per observation"
        ),
    )
    ll = logdensity(model.prior, _trajectory_state(xs, 0))
    for t in eachindex(ys)
        x = _trajectory_state(xs, t)
        ll += logdensity(model.dyn, t, _trajectory_state(xs, t - 1), x)
        ll += logdensity(model.obs, t, x, ys[t])
    end
    return ll
end
