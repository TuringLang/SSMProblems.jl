import Distributions: ContinuousMultivariateDistribution, _logpdf, _rand!
using Bijectors: Bijectors
import Bijectors: bijector

export SSMTrajectory

## SSMTrajectory DISTRIBUTION ####################################################################

# NOTE: ContinuousMultivariateDistribution is a simplification. For discrete trajectories
# (HMMs), this would need to be generalized.

"""
    SSMTrajectory{MT, AFT, YT} <: ContinuousMultivariateDistribution

A distribution over state-space model trajectories. Used in Turing `@model` blocks to mark
the trajectory variable for `ParticleGibbs`:

```julia
# Regular SSM (no inner filter needed)
x ~ SSMTrajectory(ssm, ys)

# HierarchicalSSM (inner analytical filter required for logpdf)
x ~ SSMTrajectory(ssm, KF(), ys)
```

The `logpdf` computes the joint log-density of the trajectory and observations:
- Regular SSM: `log p(x₀) + Σ_t [log p(xₜ|xₜ₋₁) + log p(yₜ|xₜ)]`
- HierarchicalSSM: outer transitions + inner marginal likelihood via `af`

The inner analytical filter `af` must match the filter used in the `RBPF` within the
`ParticleGibbs` sampler (e.g., both `KF()` or both `KF(jitter=1e-8)`).
"""
struct SSMTrajectory{MT<:AbstractStateSpaceModel,AFT,YT} <:
       ContinuousMultivariateDistribution
    model::MT
    af::AFT
    observations::YT
end

# Convenience constructor for regular SSMs (no inner analytical filter needed).
function SSMTrajectory(model::AbstractStateSpaceModel, observations)
    SSMTrajectory(model, nothing, observations)
end

bijector(::SSMTrajectory) = Bijectors.Identity{1}()

## DIMENSIONS ####################################################################################

function _state_dim(d::SSMTrajectory{<:StateSpaceModel})
    length(SSMProblems.distribution(prior(d.model)))
end
function _state_dim(d::SSMTrajectory{<:HierarchicalSSM})
    return length(SSMProblems.distribution(d.model.outer_prior))
end

Base.length(d::SSMTrajectory) = (length(d.observations) + 1) * _state_dim(d)
Base.eltype(::Type{<:SSMTrajectory}) = Float64

## FLATTEN / UNFLATTEN ###########################################################################

function _unflatten_trajectory(x_flat::AbstractVector, T::Integer, Dx::Integer)
    states = [x_flat[((i * Dx) + 1):((i + 1) * Dx)] for i in 0:T]
    return OffsetVector(states, -1)
end

function _flatten_trajectory(traj, T::Integer, Dx::Integer)
    x_flat = Vector{Float64}(undef, (T + 1) * Dx)
    for i in 0:T
        x_flat[((i * Dx) + 1):((i + 1) * Dx)] = traj[i]
    end
    return x_flat
end

## LOGPDF ########################################################################################

# These methods inline the trajectory_logdensity computation with 1-based indexing to avoid
# constructing an OffsetVector, which Zygote cannot differentiate through.

function _logpdf(d::SSMTrajectory{<:StateSpaceModel}, x_flat::AbstractVector{<:Real})
    T = length(d.observations)
    Dx = _state_dim(d)
    # 1-indexed: states[1] = x₀, states[t+1] = xₜ
    states = [x_flat[((i - 1) * Dx + 1):(i * Dx)] for i in 1:(T + 1)]

    m = d.model
    ll = logpdf(SSMProblems.distribution(prior(m)), states[1])
    for t in 1:T
        ll += SSMProblems.logdensity(dyn(m), t, states[t], states[t + 1])
        ll += SSMProblems.logdensity(obs(m), t, states[t + 1], d.observations[t])
    end
    return ll
end

function _logpdf(d::SSMTrajectory{<:HierarchicalSSM}, x_flat::AbstractVector{<:Real})
    T = length(d.observations)
    Dx = _state_dim(d)
    # 1-indexed: states[1] = u₀, states[t+1] = uₜ
    states = [x_flat[((i - 1) * Dx + 1):(i * Dx)] for i in 1:(T + 1)]

    m = d.model
    ll = logpdf(SSMProblems.distribution(m.outer_prior), states[1])
    for t in 1:T
        ll += SSMProblems.logdensity(m.outer_dyn, t, states[t], states[t + 1])
    end

    ll += inner_loglikelihood(d.af, m.inner_model, states, d.observations)
    return ll
end

## RAND ##########################################################################################

function _rand!(
    rng::AbstractRNG, d::SSMTrajectory{<:StateSpaceModel}, x::AbstractVector{<:Real}
)
    T = length(d.observations)
    Dx = _state_dim(d)
    x0, xs, _ = SSMProblems.sample(rng, d.model, T)
    x[1:Dx] = x0
    for t in 1:T
        x[((t * Dx) + 1):((t + 1) * Dx)] = xs[t]
    end
    return x
end

function _rand!(
    rng::AbstractRNG, d::SSMTrajectory{<:HierarchicalSSM}, x::AbstractVector{<:Real}
)
    T = length(d.observations)
    Dx = _state_dim(d)
    x0, _, xs, _, _ = SSMProblems.sample(rng, d.model, T)
    x[1:Dx] = x0
    for t in 1:T
        x[((t * Dx) + 1):((t + 1) * Dx)] = xs[t]
    end
    return x
end
