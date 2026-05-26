using LogDensityProblems: LogDensityProblems
import Distributions: logpdf

export ssm_loglikelihood, trajectory_logdensity
export ParameterisedSSM, SSMParameterLogDensity

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
    ll += ssm_loglikelihood(af, model.inner_model, nothing, observations; controls=controls)

    return ll
end

## PARAMETERISED SSM ###########################################################################

"""
    ParameterisedSSM(build, observations)

A parameterised state-space model that maps parameter vectors to concrete SSMs.

# Fields
- `build`: A callable `θ -> AbstractStateSpaceModel` that constructs an SSM from parameters.
  Fixed model components should be captured via closure.
- `observations`: The observation sequence y₁:T.

# Example
```julia
function build_model(θ, fixed)
    b = θ[1:2]
    dyn = LinearGaussianLatentDynamics(fixed.A, b, fixed.Q)
    return StateSpaceModel(fixed.prior, dyn, fixed.obs)
end

pssm = ParameterisedSSM(θ -> build_model(θ, fixed), observations)
model = pssm.build(θ)  # returns a concrete SSM
```
"""
struct ParameterisedSSM{F,YT}
    build::F
    observations::YT
end

## SSM PARAMETER LOG-DENSITY ###################################################################

"""
    SSMParameterLogDensity(prior, param_model, af, trajectory)
    SSMParameterLogDensity(prior, param_model, trajectory)

Log-density for SSM parameters θ conditioned on a fixed trajectory:

    log p(θ | trajectory, y) ∝ log p(θ) + log p(trajectory, y | θ)

Implements the `LogDensityProblems` interface.

# Fields
- `prior`: Prior distribution on θ (any Distributions.jl distribution)
- `param_model`: A `ParameterisedSSM` mapping θ to an SSM
- `af`: Inner analytical filter for HierarchicalSSM (e.g., `KalmanFilter()`), or `nothing`
  for regular SSMs
- `trajectory`: Current reference trajectory ([`ReferenceTrajectory`](@ref) indexed from 0)
"""
struct SSMParameterLogDensity{PT,MT<:ParameterisedSSM,AFT,TT}
    prior::PT
    param_model::MT
    af::AFT
    trajectory::TT
end

function SSMParameterLogDensity(prior, param_model::ParameterisedSSM, trajectory)
    return SSMParameterLogDensity(prior, param_model, nothing, trajectory)
end

function LogDensityProblems.capabilities(::Type{<:SSMParameterLogDensity})
    return LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.dimension(ld::SSMParameterLogDensity)
    return length(ld.prior)
end

function LogDensityProblems.logdensity(ld::SSMParameterLogDensity{<:Any,<:Any,Nothing}, θ)
    model = ld.param_model.build(θ)
    return logpdf(ld.prior, θ) +
           trajectory_logdensity(model, ld.trajectory, ld.param_model.observations)
end

function LogDensityProblems.logdensity(
    ld::SSMParameterLogDensity{<:Any,<:Any,<:AbstractFilter}, θ
)
    model = ld.param_model.build(θ)
    return logpdf(ld.prior, θ) +
           trajectory_logdensity(model, ld.af, ld.trajectory, ld.param_model.observations)
end
