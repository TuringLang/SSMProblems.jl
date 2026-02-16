import LogExpFunctions: softmax
import SSMProblems: prior, dyn

export ConditionalSMC, CSMC, CSMCAS
export CSMCModel, CSMCState

## TRAJECTORY REFRESHMENT STRATEGIES #######################################################

abstract type AbstractTrajectoryRefreshment end

"""
    NoRefreshment <: AbstractTrajectoryRefreshment

Vanilla conditional SMC with no trajectory refreshment. Uses `filter()` directly with a
reference trajectory pinned to particle 1.
"""
struct NoRefreshment <: AbstractTrajectoryRefreshment end

"""
    AncestorSampling <: AbstractTrajectoryRefreshment

Conditional SMC with ancestor sampling (CSMC-AS / PGAS). At each time step, the reference
particle's ancestor is resampled using backward weights, improving mixing for the full
trajectory. For RBPF models, backward predictive likelihoods are computed at the start of
each sweep which enable closed form ancestor weights.
"""
struct AncestorSampling <: AbstractTrajectoryRefreshment end

## CSMC SAMPLER ############################################################################

"""
    ConditionalSMC{PF, TR} <: AbstractMCMC.AbstractSampler

Conditional Sequential Monte Carlo sampler with configurable trajectory refreshment.

# Fields
- `pf::PF`: The underlying particle filter (e.g., `BF(N)`, `RBPF(BF(N), KF())`)
- `refreshment::TR`: Trajectory refreshment strategy

# Examples
```julia
ConditionalSMC(BF(100), AncestorSampling())

# Shorthand constructors
CSMC(BF(100))                           # Vanilla CSMC
CSMCAS(BF(100))                         # CSMC with ancestor sampling
CSMCAS(RBPF(BF(200), KF()))             # Rao-Blackwellised PGAS
```
"""
struct ConditionalSMC{PF<:AbstractParticleFilter,TR<:AbstractTrajectoryRefreshment} <:
       AbstractMCMC.AbstractSampler
    pf::PF
    refreshment::TR
end

CSMC(pf) = ConditionalSMC(pf, NoRefreshment())
CSMCAS(pf) = ConditionalSMC(pf, AncestorSampling())

## STATE AND MODEL #########################################################################

"""
    CSMCState{TT}

State of a conditional SMC sampler, containing the current reference trajectory.

The trajectory is an `OffsetVector` indexed from 0 (matching the prior at time 0).
For RBPF, the trajectory contains `RBState` objects (outer state + inner filtering
distribution).
"""
struct CSMCState{TT}
    trajectory::TT
end

"""
    CSMCModel{MT, YT} <: AbstractMCMC.AbstractModel

Model wrapper for standalone CSMC sampling via the AbstractMCMC interface.

# Fields
- `ssm::MT`: The state-space model
- `observations::YT`: Vector of observations
"""
struct CSMCModel{MT<:AbstractStateSpaceModel,YT<:AbstractVector} <:
       AbstractMCMC.AbstractModel
    ssm::MT
    observations::YT
end

## REF_STATE EXTRACTION ####################################################################

# CSMCState stores full trajectories (RBState for RBPF). The filter/initialise/move
# functions expect ref_state to contain only outer states for RBPF. _make_ref_state
# handles this conversion.
# TODO: is this actually needed?
_make_ref_state(::Nothing) = nothing
_make_ref_state(traj) = traj
_make_ref_state(traj::OffsetVector{<:RBState}) = getproperty.(traj, :x)

## TRAJECTORY SAMPLING #####################################################################

function _sample_trajectory(
    rng::AbstractRNG, container::DenseParticleContainer, state::ParticleDistribution
)
    ws = get_weights(state)
    idx = randcat(rng, ws)
    return get_ancestry(container, idx)
end

## BACKWARD PREDICTIVE LIKELIHOODS #########################################################

# Default: no backward likelihoods needed (regular PF, or first iteration)
_compute_backward_likelihoods(rng, model, pf, observations, ref_state) = nothing

_backward_predictor(::KalmanFilter) = BackwardInformationPredictor(; initial_jitter=1e-8)
_backward_predictor(::DiscreteFilter) = BackwardDiscretePredictor()

_backward_init_kwargs(::HierarchicalSSM, ::KalmanFilter) = (;)
function _backward_init_kwargs(model::HierarchicalSSM, ::DiscreteFilter)
    return (; num_states=length(calc_α0(model.inner_model.prior)))
end

function _compute_backward_likelihoods(
    rng::AbstractRNG, model::HierarchicalSSM, pf::RBPF, observations, ref_state
)
    isnothing(ref_state) && return nothing
    K = length(observations)
    inner = model.inner_model
    bp = _backward_predictor(pf.af)
    init_kw = _backward_init_kwargs(model, pf.af)

    pred_lik = backward_initialise(
        rng, inner.obs, bp, K, observations[K]; new_outer=ref_state[K], init_kw...
    )
    liks = Vector{typeof(pred_lik)}(undef, K)
    liks[K] = pred_lik
    for t in (K - 1):-1:1
        pred_lik = backward_predict(
            rng,
            inner.dyn,
            bp,
            t,
            pred_lik;
            prev_outer=ref_state[t],
            new_outer=ref_state[t + 1],
        )
        pred_lik = backward_update(
            inner.obs, bp, t, pred_lik, observations[t]; new_outer=ref_state[t]
        )
        liks[t] = pred_lik
    end
    return liks
end

## ANCESTOR SAMPLING HELPERS ###############################################################

# Regular PF: ref state for ancestor weight is just the trajectory state
_build_ancestor_ref(ref_state, ::Nothing, t) = ref_state[t]

# RBPF: ref state for ancestor weight is RBState(outer_state, backward_likelihood)
function _build_ancestor_ref(ref_state, back_liks::AbstractVector, t)
    return RBState(ref_state[t], back_liks[t])
end

## CSMC IMPLEMENTATIONS ###################################################################

"""
    _csmc_sample(rng, model, csmc, observations, ref_traj)

Run one conditional SMC sweep, returning `(trajectory, log_likelihood)`.

`ref_traj` is the reference trajectory from the previous iteration (or `nothing` for
the initial unconditional run). For RBPF, this may contain `RBState` objects; outer
states are extracted automatically via `_make_ref_state`.
"""
function _csmc_sample(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    csmc::ConditionalSMC{<:Any,NoRefreshment},
    observations,
    ref_traj,
)
    ref_state = _make_ref_state(ref_traj)
    cb = DenseAncestorCallback(nothing)
    state, ll = filter(rng, model, csmc.pf, observations; ref_state, callback=cb)
    trajectory = _sample_trajectory(rng, cb.container, state)
    return trajectory, ll
end

function _csmc_sample(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    csmc::ConditionalSMC{<:Any,AncestorSampling},
    observations,
    ref_traj,
)
    pf = csmc.pf
    K = length(observations)
    ref_state = _make_ref_state(ref_traj)

    # Backward predictive likelihoods (only non-nothing for RBPF)
    back_liks = _compute_backward_likelihoods(rng, model, pf, observations, ref_state)

    cb = DenseAncestorCallback(nothing)
    state = initialise(rng, prior(model), pf; ref_state)
    cb(model, pf, state, observations, PostInit)

    for t in 1:K
        # Ancestor sampling for the reference particle
        if !isnothing(ref_state)
            ref_as = _build_ancestor_ref(ref_state, back_liks, t)
            as_weights = map(state.particles) do particle
                ancestor_weight(particle, dyn(model), pf, t, ref_as)
            end
            ancestor_idx = randcat(rng, softmax(as_weights))
        end

        state = resample(rng, resampler(pf), state; ref_state)

        if !isnothing(ref_state)
            state.particles[1] = Particle(
                state.particles[1].state, state.particles[1].log_w, ancestor_idx
            )
        end

        state, _ = move(rng, model, pf, t, state, observations[t]; ref_state)

        cb(model, pf, t, state, observations[t], PostUpdate)
    end

    trajectory = _sample_trajectory(rng, cb.container, state)
    return trajectory, zero(eltype(log_weights(state)))
end

## ABSTRACTMCMC INTERFACE ##################################################################

function AbstractMCMC.step(
    rng::AbstractRNG, model::CSMCModel, csmc::ConditionalSMC; kwargs...
)
    traj, ll = _csmc_sample(rng, model.ssm, csmc, model.observations, nothing)
    return CSMCState(traj), CSMCState(traj)
end

function AbstractMCMC.step(
    rng::AbstractRNG, model::CSMCModel, csmc::ConditionalSMC, state::CSMCState; kwargs...
)
    traj, ll = _csmc_sample(rng, model.ssm, csmc, model.observations, state.trajectory)
    return CSMCState(traj), CSMCState(traj)
end
