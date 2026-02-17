using LogDensityProblems: LogDensityProblems
import Distributions: logpdf

export trajectory_logdensity, inner_loglikelihood, kf_loglikelihood
export ParameterisedSSM, SSMParameterLogDensity

## TRAJECTORY LOG-DENSITY ######################################################################

"""
    trajectory_logdensity(model::StateSpaceModel, trajectory, observations)

Compute the joint log-density of a trajectory and observations under a regular SSM:

    log p(x₀) + Σ_t [log p(xₜ | xₜ₋₁) + log p(yₜ | xₜ)]

The `trajectory` should be an OffsetVector indexed from 0 (matching the prior at time 0).
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
trajectory, computed by `inner_loglikelihood` using the analytical filter `af`.

The `outer_trajectory` should be an OffsetVector indexed from 0.
"""
function trajectory_logdensity(
    model::HierarchicalSSM,
    af::AbstractFilter,
    outer_trajectory,
    observations::AbstractVector,
)
    T = length(observations)

    # Outer prior + transitions
    ll = logpdf(SSMProblems.distribution(model.outer_prior), outer_trajectory[0])
    for t in 1:T
        ll += SSMProblems.logdensity(
            model.outer_dyn, t, outer_trajectory[t - 1], outer_trajectory[t]
        )
    end

    # Inner marginal log-likelihood via analytical filter
    ll += inner_loglikelihood(af, model.inner_model, outer_trajectory, observations)

    return ll
end

## INNER LOG-LIKELIHOOD ########################################################################

"""
    inner_loglikelihood(af::AbstractFilter, inner_model, outer_trajectory, observations)

Compute the marginal log-likelihood log p(y₁:T | u₀:T) of the inner model conditioned on
the outer trajectory, using the analytical filter `af`.

Dispatches on the filter type to select the appropriate algorithm.
"""
function inner_loglikelihood end

"""
    inner_loglikelihood(af::KalmanFilter, inner_model, outer_trajectory, observations)

KalmanFilter specialization: extracts linear-Gaussian parameters at each timestep and
delegates to `kf_loglikelihood`.
"""
function inner_loglikelihood(
    ::KalmanFilter,
    inner_model::StateSpaceModel,
    outer_trajectory,
    observations::AbstractVector,
)
    T = length(observations)
    inner_dyn = inner_model.dyn
    inner_obs = inner_model.obs
    inner_pr = inner_model.prior

    μ0 = calc_μ0(inner_pr; new_outer=outer_trajectory[0])
    Σ0 = calc_Σ0(inner_pr; new_outer=outer_trajectory[0])

    As = map(1:T) do t
        calc_A(inner_dyn, t; prev_outer=outer_trajectory[t - 1], new_outer=outer_trajectory[t])
    end
    bs = map(1:T) do t
        calc_b(inner_dyn, t; prev_outer=outer_trajectory[t - 1], new_outer=outer_trajectory[t])
    end
    Qs = map(1:T) do t
        calc_Q(inner_dyn, t; prev_outer=outer_trajectory[t - 1], new_outer=outer_trajectory[t])
    end
    Hs = map(1:T) do t
        calc_H(inner_obs, t; new_outer=outer_trajectory[t])
    end
    cs = map(1:T) do t
        calc_c(inner_obs, t; new_outer=outer_trajectory[t])
    end
    Rs = map(1:T) do t
        calc_R(inner_obs, t; new_outer=outer_trajectory[t])
    end

    return kf_loglikelihood(μ0, Σ0, As, bs, Qs, Hs, cs, Rs, observations)
end

## KF LOG-LIKELIHOOD ###########################################################################

"""
    kf_loglikelihood(μ0, Σ0, As, bs, Qs, Hs, cs, Rs, ys)

Compute the marginal log-likelihood of observations under a linear-Gaussian model via the
Kalman filter forward pass.

Accepts PDMat natively for `Σ0`, `Qs`, `Rs`. A `ChainRulesCore.rrule` is registered for this
function to enable efficient reverse-mode AD gradients using the analytical backward recursion
from `kalman_gradient.jl`.

# Arguments
- `μ0`: Initial mean vector
- `Σ0`: Initial covariance (AbstractPDMat or AbstractMatrix)
- `As`: Vector of transition matrices, one per timestep
- `bs`: Vector of transition offsets, one per timestep
- `Qs`: Vector of process noise covariances, one per timestep
- `Hs`: Vector of observation matrices, one per timestep
- `cs`: Vector of observation offsets, one per timestep
- `Rs`: Vector of observation noise covariances, one per timestep
- `ys`: Vector of observations

# Returns
Total log-likelihood: log p(y₁:T) = Σ_t log p(yₜ | y₁:ₜ₋₁)
"""
function kf_loglikelihood(μ0, Σ0, As, bs, Qs, Hs, cs, Rs, ys)
    T = length(ys)
    state = MvNormal(μ0, Σ0)
    ll = zero(eltype(μ0))

    for t in 1:T
        # Predict
        μ, Σ = params(state)
        μ̂ = As[t] * μ + bs[t]
        Σ̂ = X_A_Xt(Σ, As[t]) + Qs[t]
        state = MvNormal(μ̂, Σ̂)

        # Update
        state, ll_inc = kalman_update(state, (Hs[t], cs[t], Rs[t]), ys[t], nothing)
        ll += ll_inc
    end

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
    dyn = HomogeneousLinearGaussianLatentDynamics(fixed.A, b, fixed.Q)
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

Implements the `LogDensityProblems` interface. The `trajectory` field should be a `Ref` so
it can be mutated between Gibbs iterations.

# Fields
- `prior`: Prior distribution on θ (any Distributions.jl distribution)
- `param_model`: A `ParameterisedSSM` mapping θ to an SSM
- `af`: Inner analytical filter for HierarchicalSSM (e.g., `KalmanFilter()`), or `nothing`
  for regular SSMs
- `trajectory`: A `Ref` holding the current reference trajectory (OffsetVector indexed from 0)
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
    LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.dimension(ld::SSMParameterLogDensity)
    return length(ld.prior)
end

function LogDensityProblems.logdensity(ld::SSMParameterLogDensity{<:Any,<:Any,Nothing}, θ)
    model = ld.param_model.build(θ)
    return logpdf(ld.prior, θ) +
           trajectory_logdensity(model, ld.trajectory[], ld.param_model.observations)
end

function LogDensityProblems.logdensity(
    ld::SSMParameterLogDensity{<:Any,<:Any,<:AbstractFilter}, θ
)
    model = ld.param_model.build(θ)
    return logpdf(ld.prior, θ) +
           trajectory_logdensity(model, ld.af, ld.trajectory[], ld.param_model.observations)
end
