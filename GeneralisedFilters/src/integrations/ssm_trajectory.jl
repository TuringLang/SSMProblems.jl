import Distributions: ContinuousMultivariateDistribution, _logpdf, _rand!

export SSMTrajectory

## SSMTrajectory DISTRIBUTION ####################################################################

# NOTE: ContinuousMultivariateDistribution is a simplification. For discrete trajectories
# (HMMs), this would need to be generalized.

"""
    SSMTrajectory{MT, AFT, YT} <: ContinuousMultivariateDistribution

A distribution over state-space model trajectories. Used in Turing `@model` blocks to mark
the trajectory variable for `ParticleGibbs`:

```julia
# Regular SSM
x ~ SSMTrajectory(ssm, ys)

# HierarchicalSSM with RBPF: stores only outer states, inner marginalised via af
x ~ SSMTrajectory(ssm, KF(), ys)

# HierarchicalSSM with plain BF: stores full (outer, inner) state at every step
x ~ SSMTrajectory(ssm, ys)
```

The `logpdf` computes the joint log-density of the trajectory and observations:
- Regular SSM / HierarchicalSSM (BF): full joint `log p(x₀) + Σ_t [log p(xₜ|xₜ₋₁) + log p(yₜ|xₜ)]`
- HierarchicalSSM (RBPF, `af` provided): outer transitions + inner marginal likelihood via `af`

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
    return SSMTrajectory(model, nothing, observations)
end

## DIMENSIONS ####################################################################################

function _state_dim(d::SSMTrajectory{<:StateSpaceModel})
    return length(SSMProblems.distribution(prior(d.model)))
end
# RBPF: trajectory stores only outer states
function _state_dim(d::SSMTrajectory{<:HierarchicalSSM})
    return length(SSMProblems.distribution(d.model.outer_prior))
end
# BF: trajectory stores the full (outer, inner) HierarchicalState
function _state_dim(d::SSMTrajectory{<:HierarchicalSSM,Nothing})
    D_outer = length(SSMProblems.distribution(d.model.outer_prior))
    D_inner = length(SSMProblems.distribution(d.model.inner_model.prior))
    return D_outer + D_inner
end

Base.length(d::SSMTrajectory) = (length(d.observations) + 1) * _state_dim(d)
Base.eltype(::Type{<:SSMTrajectory}) = Float64

## FLATTEN / UNFLATTEN ###########################################################################

function _unflatten_trajectory(x_flat::AbstractVector, T::Integer, Dx::Integer)
    states = [x_flat[((i * Dx) + 1):((i + 1) * Dx)] for i in 0:T]
    return OffsetVector(states, -1)
end

function _flatten_trajectory(traj, T::Integer, Dx::Integer)
    x_flat = Vector{eltype(first(traj))}(undef, (T + 1) * Dx)
    for i in 0:T
        x_flat[((i * Dx) + 1):((i + 1) * Dx)] = traj[i]
    end
    return x_flat
end

function _flatten_trajectory(
    traj::AbstractVector{<:HierarchicalState}, T::Integer, Dx::Integer
)
    D_outer = length(first(traj).x)
    x_flat = Vector{eltype(first(traj).x)}(undef, (T + 1) * Dx)
    for i in 0:T
        x_flat[(i * Dx + 1):(i * Dx + D_outer)] = traj[i].x
        x_flat[(i * Dx + D_outer + 1):((i + 1) * Dx)] = traj[i].z
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

# BF path: trajectory stores the full (outer, inner) state at each step. The flat vector
# interleaves outer and inner components: [x₀; z₀; x₁; z₁; …; xₜ; zₜ].
function _logpdf(
    d::SSMTrajectory{<:HierarchicalSSM,Nothing}, x_flat::AbstractVector{<:Real}
)
    T = length(d.observations)
    m = d.model
    D_outer = length(SSMProblems.distribution(m.outer_prior))
    Dx = _state_dim(d)  # D_outer + D_inner
    # 1-indexed slices; states[1] = time 0, states[t+1] = time t
    xs = [x_flat[((i - 1) * Dx + 1):((i - 1) * Dx + D_outer)] for i in 1:(T + 1)]
    zs = [x_flat[((i - 1) * Dx + D_outer + 1):(i * Dx)] for i in 1:(T + 1)]

    ll = logpdf(SSMProblems.distribution(m.outer_prior), xs[1])
    ll += logpdf(SSMProblems.distribution(m.inner_model.prior; new_outer=xs[1]), zs[1])
    for t in 1:T
        ll += SSMProblems.logdensity(m.outer_dyn, t, xs[t], xs[t + 1])
        ll += SSMProblems.logdensity(
            m.inner_model.dyn, t, zs[t], zs[t + 1]; prev_outer=xs[t], new_outer=xs[t + 1]
        )
        ll += SSMProblems.logdensity(
            m.inner_model.obs, t, zs[t + 1], d.observations[t]; new_outer=xs[t + 1]
        )
    end
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

function _rand!(
    rng::AbstractRNG, d::SSMTrajectory{<:HierarchicalSSM,Nothing}, x::AbstractVector{<:Real}
)
    T = length(d.observations)
    D_outer = length(SSMProblems.distribution(d.model.outer_prior))
    Dx = _state_dim(d)
    x0, z0, xs, zs, _ = SSMProblems.sample(rng, d.model, T)
    x[1:D_outer] = x0
    x[(D_outer + 1):Dx] = z0
    for t in 1:T
        x[(t * Dx + 1):(t * Dx + D_outer)] = xs[t]
        x[(t * Dx + D_outer + 1):((t + 1) * Dx)] = zs[t]
    end
    return x
end
