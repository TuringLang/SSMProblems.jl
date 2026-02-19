import LogExpFunctions: softmax
import SSMProblems: prior, dyn

export ConditionalSMC, CSMC, CSMCBS, CSMCAS
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

"""
    BackwardSimulation <: AbstractTrajectoryRefreshment

Conditional SMC with backward simulation (CSMC-BS). Runs a full forward filter with particle
storage, then samples a trajectory via a backward pass using backward sampling weights.
For RBPF models, backward predictive likelihoods are computed on-the-fly during the backward
pass.

Note: requires O(N*T) storage since the full particle history must be retained.
"""
struct BackwardSimulation <: AbstractTrajectoryRefreshment end

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
CSMCBS(pf) = ConditionalSMC(pf, BackwardSimulation())
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

function _sample_trajectory(
    rng::AbstractRNG, tree::ParticleTree, state::ParticleDistribution
)
    ws = get_weights(state)
    path = rand(rng, tree, ws)
    return OffsetVector(path, -1)
end

## PARTICLE TREE HELPERS ###################################################################

function _init_tree(state::ParticleDistribution)
    states = Vector(getfield.(state.particles, :state))
    N = length(states)
    return ParticleTree(states, floor(Int64, N * log(N)))
end

function _update_tree!(tree::ParticleTree, state::ParticleDistribution)
    particles = state.particles
    ancestors = Vector{Int64}(getfield.(particles, :ancestor))
    states = Vector(getfield.(particles, :state))
    prune!(tree, get_offspring(ancestors))
    insert!(tree, states, ancestors)
    return tree
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
    pf = csmc.pf
    K = length(observations)
    ref_state = _make_ref_state(ref_traj)

    state = initialise(rng, prior(model), pf; ref_state)
    tree = _init_tree(state)

    state, ll = step(rng, model, pf, 1, state, observations[1]; ref_state)
    _update_tree!(tree, state)

    for t in 2:K
        state, ll_inc = step(rng, model, pf, t, state, observations[t]; ref_state)
        ll += ll_inc
        _update_tree!(tree, state)
    end

    trajectory = _sample_trajectory(rng, tree, state)
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

    state = initialise(rng, prior(model), pf; ref_state)
    tree = _init_tree(state)

    ll = zero(eltype(log_weights(state)))
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

        state, ll_inc = move(rng, model, pf, t, state, observations[t]; ref_state)
        ll += ll_inc

        _update_tree!(tree, state)
    end

    trajectory = _sample_trajectory(rng, tree, state)
    return trajectory, ll
end

## BACKWARD SIMULATION HELPERS #############################################################

# Initialize backward predictive likelihood at time K (no-op for non-RBPF)
_bs_init_back_lik(rng, model, pf, observations, K, state_K) = nothing

function _bs_init_back_lik(
    rng::AbstractRNG,
    model::HierarchicalSSM,
    pf::RBPF,
    observations,
    K::Integer,
    state_K::RBState,
)
    bp = _backward_predictor(pf.af)
    init_kw = _backward_init_kwargs(model, pf.af)
    return backward_initialise(
        rng, model.inner_model.obs, bp, K, observations[K]; new_outer=state_K.x, init_kw...
    )
end

# Build reference state for backward weights (combines state with backward likelihood)
_build_bs_ref(state, ::Nothing) = state
_build_bs_ref(state::RBState, back_lik) = RBState(state.x, back_lik)

# Update backward predictive likelihood during backward pass (no-op for non-RBPF)
function _bs_step_back_lik(
    rng, model, pf, t, ::Nothing, observations, prev_state, next_state
)
    nothing
end

function _bs_step_back_lik(
    rng::AbstractRNG,
    model::HierarchicalSSM,
    pf::RBPF,
    t::Integer,
    back_lik,
    observations,
    prev_state::RBState,
    next_state::RBState,
)
    bp = _backward_predictor(pf.af)
    pred_lik = backward_predict(
        rng,
        model.inner_model.dyn,
        bp,
        t,
        back_lik;
        prev_outer=prev_state.x,
        new_outer=next_state.x,
    )
    return backward_update(
        model.inner_model.obs, bp, t, pred_lik, observations[t]; new_outer=prev_state.x
    )
end

function _csmc_sample(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    csmc::ConditionalSMC{<:Any,BackwardSimulation},
    observations,
    ref_traj,
)
    pf = csmc.pf
    K = length(observations)
    ref_state = _make_ref_state(ref_traj)

    # Forward filtering pass, storing particles at each timestep
    init_state = initialise(rng, prior(model), pf; ref_state)
    init_particles = deepcopy(init_state.particles)

    state, ll = step(rng, model, pf, 1, init_state, observations[1]; ref_state)
    particle_history = Vector{typeof(state.particles)}(undef, K)
    particle_history[1] = deepcopy(state.particles)

    for t in 2:K
        state, ll_inc = step(rng, model, pf, t, state, observations[t]; ref_state)
        ll += ll_inc
        particle_history[t] = deepcopy(state.particles)
    end

    # Backward simulation pass
    ws = get_weights(state)
    idx = randcat(rng, ws)
    sampled_state = particle_history[K][idx].state

    back_lik = _bs_init_back_lik(rng, model, pf, observations, K, sampled_state)

    ST = typeof(sampled_state)
    trajectory = OffsetVector(Vector{ST}(undef, K + 1), -1)
    trajectory[K] = sampled_state

    for t in (K - 1):-1:1
        ref_next = _build_bs_ref(trajectory[t + 1], back_lik)
        backward_ws = map(particle_history[t]) do particle
            ancestor_weight(particle, dyn(model), pf, t + 1, ref_next)
        end
        idx = randcat(rng, softmax(backward_ws))
        trajectory[t] = particle_history[t][idx].state

        back_lik = _bs_step_back_lik(
            rng, model, pf, t, back_lik, observations, trajectory[t], trajectory[t + 1]
        )
    end

    # Time 0: backward step from t=1 to initial particles
    ref_at_1 = _build_bs_ref(trajectory[1], back_lik)
    backward_ws = map(init_particles) do particle
        ancestor_weight(particle, dyn(model), pf, 1, ref_at_1)
    end
    idx = randcat(rng, softmax(backward_ws))
    trajectory[0] = init_particles[idx].state

    return trajectory, ll
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
