using LogDensityProblems: LogDensityProblems
import Distributions: logpdf

export ssm_loglikelihood, trajectory_logdensity
export TrajectoryParameterLogDensity

## SSM LOG-LIKELIHOOD ##########################################################################

"""
    ssm_loglikelihood(filter, model, θ, ys; controls=(;)) -> ll

Marginal log-likelihood log p(y₁:T | θ) of a state-space model under the analytical filter
`filter` (currently `KalmanFilter`). Parameter resolution is done once here: `FixedParametric`
fields are evaluated before the time loop and the resulting primal is shared across all
steps (Mooncake's tape sees a single primal with `T` consumers in the rrule path).

Each step calls the filter's `_step_forward` on already-resolved parameter NamedTuples.

# Arguments
- `filter`: An analytical filter (e.g. [`KalmanFilter`](@ref)).
- `model`: A state-space model whose components are parameter-wrapped.
- `θ`: Inference parameters; passed to `FixedParametric` / `TimeVaryingParametric` closures.
- `ys`: Observation sequence.

# Keyword arguments
- `controls`: A `NamedTuple` of [`AbstractModelParameter`](@ref)-typed control entries
  (e.g. `prev_outer`, `new_outer` for hierarchical models; or shared θ-dependent helpers).
"""
function ssm_loglikelihood(
    filter::AbstractFilter,
    model::AbstractStateSpaceModel,
    θ,
    ys::AbstractVector;
    controls::NamedTuple=(;),
)
    return _ssm_loglikelihood(filter, model, θ, ys, controls)
end

# Positional helper that the Mooncake rrule!! is registered against (rrules are easier
# to write against positional-only functions than over kwarg-lowered methods).
function _ssm_loglikelihood(filter, model, θ, ys, controls)
    hoisted_controls = hoist_controls(controls, θ)

    prior_hoist = hoist_static(prior(model), θ, hoisted_controls)
    dyn_hoist = hoist_static(dyn(model), θ, hoisted_controls)
    obs_hoist = hoist_static(obs(model), θ, hoisted_controls)

    prior_params = step_params(prior(model), θ, hoisted_controls, prior_hoist)
    state = _step_initial(filter, prior_params)

    ll = zero(eltype(eltype(ys)))
    for t in eachindex(ys)
        resolved = resolve_controls(controls, hoisted_controls, θ, t)
        dyn_params = step_params(dyn(model), θ, t, resolved, dyn_hoist)
        obs_params = step_params(obs(model), θ, t, resolved, obs_hoist)
        state, ll_inc, _ = _step_forward(filter, state, dyn_params, obs_params, ys[t])
        ll += ll_inc
    end
    return ll
end

## TRAJECTORY LOG-DENSITY ######################################################################

"""
    trajectory_logdensity(model::StateSpaceModel, trajectory, observations)

Compute the joint log-density of a trajectory and observations under a regular SSM:

    log p(x₀) + Σ_t [log p(xₜ | xₜ₋₁) + log p(yₜ | xₜ)]

The `trajectory` should be a [`ReferenceTrajectory`](@ref) indexed from 0 (matching the
prior at time 0).
"""
function trajectory_logdensity(
    model::StateSpaceModel, trajectory, observations::AbstractVector
)
    T = length(observations)
    ll = logpdf(SSMProblems.distribution(prior(model)), trajectory[0])
    for t in 1:T
        ll += SSMProblems.logdensity(dyn(model), t, trajectory[t - 1], trajectory[t])
        ll += SSMProblems.logdensity(obs(model), t, trajectory[t], observations[t])
    end
    return ll
end

"""
    trajectory_logdensity(model::HierarchicalSSM, af::AbstractFilter, outer_trajectory, observations)

Compute the joint log-density of an outer trajectory and observations under a hierarchical SSM:

    log p(u₀) + Σ_t log p(uₜ | uₜ₋₁) + log p(y₁:T | u₀:T)

The last term is the marginal log-likelihood of the inner model conditioned on the outer
trajectory, computed by [`ssm_loglikelihood`](@ref) with `prev_outer` / `new_outer`
[`TimeVarying`](@ref) controls carrying the outer states.

The `outer_trajectory` should be a [`ReferenceTrajectory`](@ref) indexed from 0.
"""
function trajectory_logdensity(
    model::HierarchicalSSM,
    af::AbstractFilter,
    outer_trajectory,
    observations::AbstractVector,
    θ=nothing,
)
    T = length(observations)

    ll = logpdf(SSMProblems.distribution(model.outer_prior), outer_trajectory[0])
    for t in 1:T
        ll += SSMProblems.logdensity(
            model.outer_dyn, t, outer_trajectory[t - 1], outer_trajectory[t]
        )
    end

    controls = (
        prev_outer=TimeVarying(t -> outer_trajectory[t - 1]),
        new_outer=TimeVarying(t -> outer_trajectory[t]),
    )
    ll += ssm_loglikelihood(af, model.inner_model, θ, observations; controls=controls)

    return ll
end

"""
    trajectory_logdensity(model::StateSpaceModel, trajectory, observations, θ)

θ-aware variant for parametric regular SSMs. `θ` is baked into the model via
[`fix`](@ref) and the resulting non-parametric model is evaluated through the standard
SSMProblems dispatches. Mooncake's auto-AD traces θ through the closure captures inside
the fixed model.
"""
function trajectory_logdensity(
    model::StateSpaceModel, trajectory, observations::AbstractVector, θ
)
    return trajectory_logdensity(fix(model, θ), trajectory, observations)
end

## TRAJECTORY PARAMETER LOG-DENSITY ############################################################

"""
    TrajectoryParameterLogDensity(prior, model, observations, trajectory)
    TrajectoryParameterLogDensity(prior, model, af, observations, trajectory)

Log-density for SSM parameters θ conditioned on a fixed trajectory:

    log p(θ | trajectory, y) ∝ log p(θ) + log p(trajectory, y | θ)

Implements the `LogDensityProblems` interface. The `model` is a single state-space model
whose components may carry [`FixedParametric`](@ref) / [`TimeVaryingParametric`](@ref)
parameters; θ is passed positionally to those closures at evaluation time.

# Fields
- `prior`: Prior distribution on θ (any Distributions.jl distribution).
- `model`: A [`StateSpaceModel`](@ref) (regular SSM) or [`HierarchicalSSM`](@ref) with
  parametric components.
- `af`: Inner analytical filter for `HierarchicalSSM` (e.g. [`KalmanFilter`](@ref)); pass
  `nothing` for regular SSMs.
- `observations`: The observation sequence y₁:T.
- `trajectory`: The reference trajectory ([`ReferenceTrajectory`](@ref) indexed from 0).
"""
struct TrajectoryParameterLogDensity{PT,MT,AFT,YT,TT}
    prior::PT
    model::MT
    af::AFT
    observations::YT
    trajectory::TT
end

function TrajectoryParameterLogDensity(prior, model::StateSpaceModel, observations, trajectory)
    return TrajectoryParameterLogDensity(prior, model, nothing, observations, trajectory)
end

function LogDensityProblems.capabilities(::Type{<:TrajectoryParameterLogDensity})
    return LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.dimension(ld::TrajectoryParameterLogDensity)
    return length(ld.prior)
end

function LogDensityProblems.logdensity(
    ld::TrajectoryParameterLogDensity{<:Any,<:StateSpaceModel,Nothing}, θ
)
    return logpdf(ld.prior, θ) +
           trajectory_logdensity(ld.model, ld.trajectory, ld.observations, θ)
end

function LogDensityProblems.logdensity(
    ld::TrajectoryParameterLogDensity{<:Any,<:HierarchicalSSM,<:AbstractFilter}, θ
)
    return logpdf(ld.prior, θ) +
           trajectory_logdensity(ld.model, ld.af, ld.trajectory, ld.observations, θ)
end
