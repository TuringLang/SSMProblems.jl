import Distributions: ContinuousMultivariateDistribution, _logpdf, _rand!
using Bijectors: Bijectors
import Bijectors: bijector

export SSMTrajectory

## SSMTrajectory DISTRIBUTION ####################################################################

# NOTE: ContinuousMultivariateDistribution is a simplification. For discrete trajectories
# (HMMs), this would need to be generalized.

"""
    SSMTrajectory{MT, YT} <: ContinuousMultivariateDistribution

A distribution over state-space model trajectories. Used in Turing `@model` blocks to mark
the trajectory variable for `ParticleGibbs`:

```julia
@model function my_model(ys, fixed)
    b ~ Normal(0, 1)
    ssm = build_ssm(b, fixed)
    x ~ SSMTrajectory(ssm, ys)
end
```

The `logpdf` computes the joint log-density of the trajectory and observations:
- Regular SSM: `log p(x₀) + Σ_t [log p(xₜ|xₜ₋₁) + log p(yₜ|xₜ)]`
- HierarchicalSSM: outer transitions + inner KF marginal likelihood

The inner analytical filter (e.g., `KF()`) is NOT stored here — it's a property of the
sampler (`ParticleGibbs`), not the trajectory. For `logpdf` computation during the parameter
step, `KF()` is auto-detected for `HierarchicalSSM` models.
"""
struct SSMTrajectory{MT<:AbstractStateSpaceModel,YT} <: ContinuousMultivariateDistribution
    model::MT
    observations::YT
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

    # Inner marginal log-likelihood via KF (inlined to avoid OffsetVector)
    inner_dyn = m.inner_model.dyn
    inner_obs = m.inner_model.obs
    inner_pr = m.inner_model.prior

    μ0 = calc_μ0(inner_pr; new_outer=states[1])
    Σ0 = calc_Σ0(inner_pr; new_outer=states[1])
    As = map(t -> calc_A(inner_dyn, t; prev_outer=states[t], new_outer=states[t + 1]), 1:T)
    bs = map(t -> calc_b(inner_dyn, t; prev_outer=states[t], new_outer=states[t + 1]), 1:T)
    Qs = map(t -> calc_Q(inner_dyn, t; prev_outer=states[t], new_outer=states[t + 1]), 1:T)
    Hs = map(t -> calc_H(inner_obs, t; new_outer=states[t + 1]), 1:T)
    cs = map(t -> calc_c(inner_obs, t; new_outer=states[t + 1]), 1:T)
    Rs = map(t -> calc_R(inner_obs, t; new_outer=states[t + 1]), 1:T)

    ll += kf_loglikelihood(μ0, Σ0, As, bs, Qs, Hs, cs, Rs, d.observations)
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
