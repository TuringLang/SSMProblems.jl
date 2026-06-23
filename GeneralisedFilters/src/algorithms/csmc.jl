import LogExpFunctions: softmax
import SSMProblems: prior, dyn

export ConditionalSMC
export CSMCModel, CSMCState
export NoRefreshment, AncestorSampling, BackwardSimulation

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
ConditionalSMC(BF(100))                                  # Vanilla CSMC (NoRefreshment default)
ConditionalSMC(BF(100), AncestorSampling())              # CSMC with ancestor sampling
ConditionalSMC(RBPF(BF(200), KF()), AncestorSampling())  # Rao-Blackwellised PGAS
```
"""
struct ConditionalSMC{PF<:AbstractParticleFilter,TR<:AbstractTrajectoryRefreshment} <:
       AbstractMCMC.AbstractSampler
    pf::PF
    refreshment::TR
end

ConditionalSMC(pf) = ConditionalSMC(pf, NoRefreshment())

## STATE AND MODEL #########################################################################

"""
    CSMCState{TT}

State of a conditional SMC sampler, containing the current reference trajectory.

The trajectory is a [`ReferenceTrajectory`](@ref) indexed from 0 (matching the prior at
time 0). For RBPF, the trajectory contains `RBState` objects (outer state + inner filtering
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
_make_ref_state(::Nothing) = nothing
_make_ref_state(traj::ReferenceTrajectory) = traj
function _make_ref_state(traj::ReferenceTrajectory{<:RBState})
    return map(s -> s.x, traj)
end

## TRAJECTORY SAMPLING #####################################################################

function _sample_trajectory(
    rng::AbstractRNG, container::DenseParticleContainer, state::ParticleDistribution
)
    ws = get_weights(state)
    idx = StatsBase.sample(rng, StatsBase.Weights(ws))
    return get_ancestry(container, idx)
end

function _sample_trajectory(
    rng::AbstractRNG, tree::ParticleTree, state::ParticleDistribution
)
    ws = get_weights(state)
    return rand(rng, tree, ws)
end

## PARTICLE TREE / CONTAINER HELPERS ######################################################

# Capacity heuristic from Jacob, Murray & Rubenthaler (2015)
_tree_capacity(N::Integer) = max(N, floor(Int64, N * log(N)))

# Construct a ParticleTree using both the time-0 and time-1 particle distributions so
# that the subsequent-state type `T` is inferred from the time-1 states (which may
# differ from the type of the initial states in Rao-Blackwellised settings).
function _init_tree(init_state::ParticleDistribution, state::ParticleDistribution)
    initial_states = map(p -> p.state, init_state.particles)
    states_t1 = map(p -> p.state, state.particles)
    ancestors_t1 = map(p -> p.ancestor, state.particles)
    return ParticleTree(
        initial_states, states_t1, ancestors_t1, _tree_capacity(length(initial_states))
    )
end

function _init_container(init_state::ParticleDistribution, state::ParticleDistribution)
    initial_states = map(p -> p.state, init_state.particles)
    return DenseParticleContainer(
        initial_states,
        map(p -> p.state, state.particles),
        Float64.(log_weights(state)),
        map(p -> p.ancestor, state.particles),
    )
end

function _update_tree!(tree::ParticleTree, state::ParticleDistribution)
    particles = state.particles
    ancestors = map(p -> p.ancestor, particles)
    states = map(p -> p.state, particles)
    prune!(tree, get_offspring(ancestors))
    insert!(tree, states, ancestors)
    return tree
end

function _update_container!(c::DenseParticleContainer, state::ParticleDistribution)
    particles = state.particles
    push!(
        c,
        map(p -> p.state, particles),
        Float64.(log_weights(state)),
        map(p -> p.ancestor, particles),
    )
    return c
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

    init_state = initialise(rng, prior(model), pf; ref_state)
    state, ll = step(rng, model, pf, 1, init_state, observations[1]; ref_state)
    tree = _init_tree(init_state, state)

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

    init_state = initialise(rng, prior(model), pf; ref_state)

    # Perform one CSMC-AS step on the current state
    function _csmc_as_step(state, t)
        ancestor_idx = 0
        if !isnothing(ref_state)
            ref_as = _build_ancestor_ref(ref_state, back_liks, t)
            as_weights = map(state.particles) do particle
                ancestor_weight(particle, dyn(model), pf, t, ref_as)
            end
            ancestor_idx = StatsBase.sample(rng, StatsBase.Weights(softmax(as_weights)))
        end

        state = resample(rng, resampler(pf), state; ref_state)

        if !isnothing(ref_state)
            state.particles[1] = Particle(
                state.particles[1].state, state.particles[1].log_w, ancestor_idx
            )
        end

        return move(rng, model, pf, t, state, observations[t]; ref_state)
    end

    state, ll = _csmc_as_step(init_state, 1)
    tree = _init_tree(init_state, state)

    for t in 2:K
        state, ll_inc = _csmc_as_step(state, t)
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
_build_bs_ref(::RBState, ::Nothing) = error("again this should error")

# Update backward predictive likelihood during backward pass (no-op for non-RBPF)
function _bs_step_back_lik(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    pf::AbstractFilter,
    t::Integer,
    ::Nothing,
    observations,
    prev_state,
    next_state,
)
    return nothing
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

function _bs_step_back_lik(
    rng::AbstractRNG,
    model::HierarchicalSSM,
    pf::RBPF,
    t::Integer,
    ::Nothing,
    observations,
    prev_state::RBState,
    next_state::RBState,
)
    return error("this should error")
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
    N = num_particles(pf)
    ref_state = _make_ref_state(ref_traj)

    # Forward filtering pass: store full history in a DenseParticleContainer.
    init_state = initialise(rng, prior(model), pf; ref_state)
    state, ll = step(rng, model, pf, 1, init_state, observations[1]; ref_state)
    container = _init_container(init_state, state)

    for t in 2:K
        state, ll_inc = step(rng, model, pf, t, state, observations[t]; ref_state)
        ll += ll_inc
        _update_container!(container, state)
    end

    # Backward simulation pass
    idx = StatsBase.sample(rng, StatsBase.Weights(get_weights(state)))
    sampled_state = container.states[K][idx]

    back_lik = _bs_init_back_lik(rng, model, pf, observations, K, sampled_state)

    xs = Vector{typeof(sampled_state)}(undef, K)
    xs[K] = sampled_state

    for t in (K - 1):-1:1
        ref_next = _build_bs_ref(xs[t + 1], back_lik)
        backward_ws = map(1:N) do i
            ancestor_weight(Particle(container, t, i), dyn(model), pf, t + 1, ref_next)
        end
        idx = StatsBase.sample(rng, StatsBase.Weights(softmax(backward_ws)))
        xs[t] = container.states[t][idx]

        back_lik = _bs_step_back_lik(
            rng, model, pf, t, back_lik, observations, xs[t], xs[t + 1]
        )
    end

    # Time 0: backward step from t=1 to initial particles.
    ref_at_1 = _build_bs_ref(xs[1], back_lik)
    backward_ws = map(init_state.particles) do particle
        ancestor_weight(particle, dyn(model), pf, 1, ref_at_1)
    end
    idx = StatsBase.sample(rng, StatsBase.Weights(softmax(backward_ws)))
    x0 = container.initial_states[idx]

    return ReferenceTrajectory(x0, xs), ll
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
