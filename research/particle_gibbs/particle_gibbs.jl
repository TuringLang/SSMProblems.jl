"""
Particle Gibbs for a linear-Gaussian SSM with unknown drift, using GeneralisedFilters'
CSMC integrated with Turing via DynamicPPL.

Model:
    x_t = a * x_{t-1} + b + ε_t,   ε_t ~ N(0, q²)
    y_t = x_t + η_t,               η_t ~ N(0, r²)
    x_0 ~ N(0, σ₀²),   b ~ N(0, σ_b²)

Ground truth is computed via an augmented-state Kalman filter where the state is [x_t, b].
"""

using Turing
using DynamicPPL
using AbstractMCMC
using GeneralisedFilters
using SSMProblems
using PDMats
using LinearAlgebra
using Random
using StableRNGs
using OffsetArrays
using StatsBase: weights as sb_weights, sample as sb_sample
using Statistics
using Distributions
using DistributionsAD
using AdvancedMH: RandomWalkProposal
# using AdvancedHMC
using Bijectors

## ── SSM Construction ─────────────────────────────────────────────────────────

function build_ssm(a, q², r², σ₀², drift)
    return create_homogeneous_linear_gaussian_model(
        [0.0], PDMat([σ₀²;;]), [a;;], [drift], PDMat([q²;;]), [1.0;;], [0.0], PDMat([r²;;])
    )
end

## ── SSMTrajectory Distribution ───────────────────────────────────────────────
#
# Wraps an SSM and observations into a "distribution" over trajectories.
# - logpdf computes the joint log p(x_{0:T}, y_{1:T} | θ).
# - rand performs forward simulation (used only for VarInfo initialisation).

struct SSMTrajectory{MT<:AbstractStateSpaceModel,YT} <:
       Distributions.ContinuousMultivariateDistribution
    model::MT
    observations::YT
end

Base.length(d::SSMTrajectory) = length(d.observations) + 1
Base.eltype(::Type{<:SSMTrajectory}) = Float64
Bijectors.bijector(::SSMTrajectory) = Bijectors.Identity{1}()

function Distributions._logpdf(d::SSMTrajectory, x::AbstractVector{<:Real})
    T = length(d.observations)
    ssm = d.model

    ll = logpdf(SSMProblems.distribution(SSMProblems.prior(ssm)), [x[1]])
    for t in 1:T
        prev = [x[t]]
        curr = [x[t + 1]]
        ll += logpdf(SSMProblems.distribution(SSMProblems.dyn(ssm), t, prev), curr)
        ll += logpdf(
            SSMProblems.distribution(SSMProblems.obs(ssm), t, curr), d.observations[t]
        )
    end
    return ll
end

function Distributions._rand!(rng::AbstractRNG, d::SSMTrajectory, x::AbstractVector{<:Real})
    T = length(d.observations)
    x0, xs, _ = SSMProblems.sample(rng, d.model, T)
    x[1] = only(x0)
    for t in 1:T
        x[t + 1] = only(xs[t])
    end
    return x
end

## ── CSMCContext ───────────────────────────────────────────────────────────────
#
# A DynamicPPL leaf context that intercepts `x ~ SSMTrajectory(...)` calls
# during model evaluation. It captures the SSMTrajectory distribution so
# the GFParticleGibbs sampler can extract the SSM and run CSMC.

struct CSMCContext <: DynamicPPL.AbstractContext
    ssm_dist::Ref{Any}
end

function DynamicPPL.tilde_assume!!(
    ctx::CSMCContext,
    dist::SSMTrajectory,
    vn::DynamicPPL.VarName,
    vi::DynamicPPL.AbstractVarInfo,
)
    ctx.ssm_dist[] = dist
    # Return current value without accumulating logpdf — the sampler will
    # re-evaluate with DefaultContext after updating the trajectory.
    x = DynamicPPL.getindex_internal(vi, vn)
    return x, vi
end

function DynamicPPL.tilde_assume!!(
    ::CSMCContext,
    dist::Distribution,
    vn::DynamicPPL.VarName,
    vi::DynamicPPL.AbstractVarInfo,
)
    return DynamicPPL.tilde_assume!!(DynamicPPL.DefaultContext(), dist, vn, vi)
end

function DynamicPPL.tilde_observe!!(
    ::CSMCContext,
    right::Distribution,
    left,
    vn::Union{DynamicPPL.VarName,Nothing},
    vi::DynamicPPL.AbstractVarInfo,
)
    return DynamicPPL.tilde_observe!!(DynamicPPL.DefaultContext(), right, left, vn, vi)
end

## ── GFParticleGibbs Sampler ──────────────────────────────────────────────────
#
# A Turing-compatible sampler that uses GeneralisedFilters' conditional SMC to
# sample latent trajectories. Intended for use in Gibbs composition:
#
#   Gibbs(:b => MH(), :x => GFParticleGibbs(BF(100)))

struct GFParticleGibbs{AT<:GeneralisedFilters.AbstractParticleFilter} <:
       AbstractMCMC.AbstractSampler
    algo::AT
end

struct GFPGState{VIT<:DynamicPPL.AbstractVarInfo,RT}
    vi::VIT
    ref_traj::RT
end

Turing.Inference.get_varinfo(state::GFPGState) = state.vi

function Turing.Inference.setparams_varinfo!!(
    model::DynamicPPL.Model,
    ::GFParticleGibbs,
    state::GFPGState,
    params::DynamicPPL.AbstractVarInfo,
)
    new_vi = last(DynamicPPL.evaluate!!(model, params))
    return GFPGState(new_vi, state.ref_traj)
end

# Convert an OffsetVector trajectory (GeneralisedFilters format) to a flat vector
# suitable for VarInfo storage (1D states only).
function _reftraj_to_flat(traj, T)
    return [only(traj[t]) for t in 0:T]
end

function _run_csmc(
    rng::AbstractRNG, spl::GFParticleGibbs, ssm_dist::SSMTrajectory, ref_traj
)
    cb = GeneralisedFilters.DenseAncestorCallback(nothing)
    pf_state, ll = GeneralisedFilters.filter(
        rng,
        ssm_dist.model,
        spl.algo,
        ssm_dist.observations;
        ref_state=ref_traj,
        callback=cb,
    )
    ws = sb_weights(pf_state)
    idx = sb_sample(rng, 1:(spl.algo.N), ws)
    new_traj = GeneralisedFilters.get_ancestry(cb.container, idx)
    return new_traj, ll
end

function _discover_ssm(model::DynamicPPL.Model, vi::DynamicPPL.AbstractVarInfo)
    ctx = CSMCContext(Ref{Any}(nothing))
    discovery_model = DynamicPPL.setleafcontext(model, ctx)
    DynamicPPL.evaluate!!(discovery_model, vi)
    return ctx.ssm_dist[]::SSMTrajectory
end

function _update_trajectory!(
    model::DynamicPPL.Model,
    vi::DynamicPPL.AbstractVarInfo,
    ref_traj,
    T::Int,
    vn::DynamicPPL.VarName,
)
    x_flat = _reftraj_to_flat(ref_traj, T)
    vi = DynamicPPL.setindex!!(vi, x_flat, vn)
    vi = last(DynamicPPL.evaluate!!(model, vi))
    return vi
end

# Initial step: the Turing wrapper creates VarInfo (which samples x from the
# SSMTrajectory via rand), then calls this.
function Turing.Inference.initialstep(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::GFParticleGibbs,
    vi::DynamicPPL.AbstractVarInfo;
    kwargs...,
)
    ssm_dist = _discover_ssm(model, vi)
    T = length(ssm_dist.observations)

    # Run unconditional PF (no reference trajectory)
    ref_traj, _ = _run_csmc(rng, spl, ssm_dist, nothing)

    # Update VarInfo with the PF trajectory and re-evaluate for correct logp
    vi = _update_trajectory!(model, vi, ref_traj, T, @varname(x))

    return DynamicPPL.ParamsWithStats(vi, model), GFPGState(vi, ref_traj)
end

# Subsequent steps: run CSMC conditioned on the reference trajectory.
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::GFParticleGibbs,
    state::GFPGState;
    kwargs...,
)
    vi = state.vi
    ssm_dist = _discover_ssm(model, vi)
    T = length(ssm_dist.observations)

    ref_traj, _ = _run_csmc(rng, spl, ssm_dist, state.ref_traj)

    vi = _update_trajectory!(model, vi, ref_traj, T, @varname(x))

    return DynamicPPL.ParamsWithStats(vi, model), GFPGState(vi, ref_traj)
end

## ── Turing Model ─────────────────────────────────────────────────────────────

@model function drift_model(ys, a, q², r², σ₀², σ_b²)
    b ~ Normal(0, sqrt(σ_b²))
    ssm = build_ssm(a, q², r², σ₀², b)
    return x ~ SSMTrajectory(ssm, ys)
end

## ── Ground Truth via Augmented-State Kalman Filter ───────────────────────────

function augmented_kf_posterior(ys, a, q², r², σ₀², σ_b²)
    A_aug = [a 1.0; 0.0 1.0]
    b_aug = [0.0, 0.0]
    Q_aug = PDMat(Symmetric([q² 0.0; 0.0 1e-12]))
    H_aug = [1.0 0.0]
    c_aug = [0.0]
    R_aug = PDMat([r²;;])
    μ0_aug = [0.0, 0.0]
    Σ0_aug = PDMat(Symmetric([σ₀² 0.0; 0.0 σ_b²]))

    aug_model = create_homogeneous_linear_gaussian_model(
        μ0_aug, Σ0_aug, A_aug, b_aug, Q_aug, H_aug, c_aug, R_aug
    )
    state, ll = GeneralisedFilters.filter(aug_model, KF(), ys)
    return state, ll
end

## ── Run ──────────────────────────────────────────────────────────────────────

rng = StableRNG(42)

# Model parameters
a = 0.8
q² = 0.1
r² = 0.5
σ₀² = 1.0
σ_b² = 4.0
T_len = 10
N_particles = 50
N_iter = 5000
N_adapt = 1000

# Generate data
true_b = 1.5
true_ssm = build_ssm(a, q², r², σ₀², true_b)
_, _, ys = SSMProblems.sample(rng, true_ssm, T_len)

# Ground truth
kf_state, _ = augmented_kf_posterior(ys, a, q², r², σ₀², σ_b²)
println("=== Augmented KF posterior of b ===")
println("  Mean: $(kf_state.μ[2])")
println("  Std:  $(sqrt(kf_state.Σ[2, 2]))")
println("  True: $true_b")

# Particle Gibbs
m = drift_model(ys, a, q², r², σ₀², σ_b²)
# b_sampler = MH(:b => RandomWalkProposal(Normal(0, 1.0)))
# b_sampler = HMC(0.1, 10)
b_sampler = NUTS(0.65)

chain = sample(
    rng,
    m,
    Gibbs(:b => b_sampler, :x => GFParticleGibbs(BF(N_particles))),
    N_iter;
    progress=true,
)

# Results
b_samples = chain[:b]
println("\n=== Particle Gibbs posterior of b ===")
println("  Mean: $(mean(b_samples))")
println("  Std:  $(std(b_samples))")
println("\n=== Ground truth ===")
println("  KF mean: $(kf_state.μ[2])")
println("  KF std:  $(sqrt(kf_state.Σ[2, 2]))")
