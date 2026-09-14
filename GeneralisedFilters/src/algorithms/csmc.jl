import LogExpFunctions: softmax

export ConditionalSMC
export CSMCModel, CSMCState
export NoRefreshment, AncestorSampling, BackwardSimulation, default_backward_predictor

## TRAJECTORY REFRESHMENT STRATEGIES #######################################################

abstract type AbstractTrajectoryRefreshment end

"""
    NoRefreshment <: AbstractTrajectoryRefreshment

Vanilla conditional SMC with no trajectory refreshment. Uses `filter()` directly with a
reference trajectory pinned to particle 1.
"""
struct NoRefreshment <: AbstractTrajectoryRefreshment end

"""
    AncestorSampling([backward_predictor]) <: AbstractTrajectoryRefreshment

Conditional SMC with ancestor sampling (CSMC-AS / PGAS). At resampling events, the reference
particle's ancestor is resampled using backward weights, improving mixing for the full
trajectory. The PF resampling trigger is respected: ancestor sampling is performed
when ordinary resampling occurs. At skipped steps, every particle retains its ancestor
and accumulated weight.
For RBPF models, backward predictive likelihoods are computed at the start of
each sweep which enable closed form ancestor weights. The optional backward predictor
uses the analytical filter's default when omitted. Gaussian defaults apply no jitter;
nonzero backward jitter is rejected because it changes the target distribution.
"""
struct AncestorSampling{BP} <: AbstractTrajectoryRefreshment
    backward_predictor::BP
end
AncestorSampling() = AncestorSampling(nothing)

"""
    BackwardSimulation([backward_predictor]) <: AbstractTrajectoryRefreshment

Conditional SMC with backward simulation (CSMC-BS). Runs a full forward filter with particle
storage, then samples a trajectory via a backward pass using backward sampling weights.
For RBPF models, backward predictive likelihoods are computed on-the-fly during the backward
pass.

Note: requires O(N*T) storage since the full particle history must be retained.
"""
struct BackwardSimulation{BP} <: AbstractTrajectoryRefreshment
    backward_predictor::BP
end
BackwardSimulation() = BackwardSimulation(nothing)

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

function _validate_csmc(model, csmc, observations, ref_traj)
    Base.require_one_based_indexing(observations)
    isempty(observations) && throw(ArgumentError("CSMC requires at least one observation"))
    rs = resampler(csmc.pf)
    supports_conditional(rs) || throw(
        ArgumentError(
            "ConditionalSMC requires a resampler with a conditional law, which " *
            "$rs does not implement. Multinomial, Systematic and Stratified " *
            "resampling are supported.",
        ),
    )
    if !isnothing(ref_traj)
        _validate_trajectory(ref_traj)
        length(ref_traj) == length(observations) + 1 || throw(
            DimensionMismatch(
                "reference trajectory must contain x0 and one state per observation"
            ),
        )
    end
    base_pf = _refreshment_filter(csmc.pf)
    if base_pf isa RBPF && !(csmc.refreshment isa NoRefreshment)
        af = base_pf.af
        af isa KalmanFilter &&
            !(af.repair isa NoRepair) &&
            throw(
                ArgumentError(
                    "Gaussian ancestor sampling/backward simulation requires KalmanFilter(repair=NoRepair())",
                ),
            )
        bp = _backward_predictor(base_pf, csmc.refreshment)
        if bp isa BackwardInformationPredictor
            any(j -> !isnothing(j) && !iszero(j), (bp.initial_jitter, bp.jitter)) && throw(
                ArgumentError(
                    "Gaussian ancestor sampling/backward simulation requires zero backward jitter; " *
                    "nonzero jitter changes the target distribution.",
                ),
            )
        end
    end
    return nothing
end

## STATE AND MODEL #########################################################################

"""
    CSMCState{TT}

State of a conditional SMC sampler, containing the current reference trajectory.

The trajectory is a [`ReferenceTrajectory`](@ref) indexed from 0 (matching the prior at
time 0). For RBPF, the trajectory contains outer states only. Inner beliefs are recomputed for
the current parameters on every sweep.
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
struct CSMCModel{MT<:StateSpaceModel,YT<:AbstractVector} <: AbstractMCMC.AbstractModel
    ssm::MT
    observations::YT
end

## REF_STATE EXTRACTION ####################################################################

# Persist outer states only: parameters change between Gibbs sweeps and invalidate beliefs.
# Accept legacy RB trajectories at the input boundary, but never return them.
_make_ref_state(::Nothing) = nothing
function _make_ref_state(traj::AbstractVector)
    return _make_ref_state(ReferenceTrajectory(first(traj), traj[2:end]))
end
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

default_backward_predictor(::KalmanFilter) = SqrtBackwardInformationPredictor()
default_backward_predictor(::SRKalmanFilter) = SqrtBackwardInformationPredictor()
default_backward_predictor(::DiscreteFilter) = BackwardDiscretePredictor()
function _backward_predictor(pf::RBPF, strategy)
    return if isnothing(strategy.backward_predictor)
        default_backward_predictor(pf.af)
    else
        strategy.backward_predictor
    end
end

function _backward_start(
    bp::Union{BackwardInformationPredictor,SqrtBackwardInformationPredictor},
    model,
    pf,
    t,
    y,
    x,
)
    return backward_initialise(bp, _component(inner_observation(model, t, x)), y)
end
function _backward_start(bp::BackwardDiscretePredictor, model, pf, t, y, x)
    n = length(_component(inner_prior(model, x)).α0)
    return backward_initialise(bp, inner_observation(model, t, x), t, y, n)
end
function _backward_observe(
    bp::Union{BackwardInformationPredictor,SqrtBackwardInformationPredictor}, lik, obs, t, y
)
    return backward_update(bp, lik, _component(obs), y)
end
function _backward_observe(bp::BackwardDiscretePredictor, lik, obs, t, y)
    return backward_update(bp, lik, obs, t, y)
end

# Only the suffix t+1:K has been initialised when a new representation is encountered.
# A small concrete union retains specialization for mixed static/dynamic models.
function _store_backward_likelihood(liks::Vector{T}, t, lik::S) where {T,S}
    if lik isa T
        liks[t] = lik
        return liks
    end
    widened = Vector{Union{T,S}}(undef, length(liks))
    for k in (t + 1):length(liks)
        widened[k] = liks[k]
    end
    widened[t] = lik
    return widened
end

_compute_backward_likelihoods(rng, model, pf, observations, ref_state, strategy) = nothing
function _compute_backward_likelihoods(
    rng::AbstractRNG, model::HierarchicalSSM, pf::RBPF, observations, ref_state, strategy
)
    isnothing(ref_state) && return nothing
    K = length(observations)
    bp = _backward_predictor(pf, strategy)
    pred_lik = _backward_start(bp, model, pf, K, observations[K], ref_state[K])
    # Preserve concrete storage on the usual homogeneous/static path. Widen only when
    # a resolved component changes the likelihood representation at an earlier time.
    liks = Vector{typeof(pred_lik)}(undef, K)
    liks[K] = pred_lik
    for t in (K - 1):-1:1
        d = _component(inner_dynamics(model, t + 1, ref_state[t], ref_state[t + 1]))
        pred_lik = backward_predict(bp, pred_lik, d)
        pred_lik = _backward_observe(
            bp, pred_lik, inner_observation(model, t, ref_state[t]), t, observations[t]
        )
        liks = _store_backward_likelihood(liks, t, pred_lik)
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
the initial unconditional run). For RBPF, the returned trajectory contains only outer states; legacy inputs containing
`RBState` objects are accepted and stripped. Ancestor sampling respects the underlying
resampling trigger and refreshes ancestors only at resampling events. All strategies require a resampler that
implements a conditional law (see `supports_conditional`).
"""
function _csmc_sample(
    rng::AbstractRNG,
    model::StateSpaceModel,
    csmc::ConditionalSMC{<:Any,NoRefreshment},
    observations,
    ref_traj,
)
    _validate_csmc(model, csmc, observations, ref_traj)
    pf = csmc.pf
    K = length(observations)
    ref_state = _make_ref_state(ref_traj)

    init_state = initialise(rng, model.prior, pf; ref_state)
    state, ll = step(rng, model, pf, 1, init_state, observations[1]; ref_state)
    tree = _init_tree(init_state, state)

    for t in 2:K
        state, ll_inc = step(rng, model, pf, t, state, observations[t]; ref_state)
        ll += ll_inc
        _update_tree!(tree, state)
    end

    trajectory = _sample_trajectory(rng, tree, state)
    return _make_ref_state(trajectory), ll
end

function _csmc_sample(
    rng::AbstractRNG,
    model::StateSpaceModel,
    csmc::ConditionalSMC{<:Any,<:AncestorSampling},
    observations,
    ref_traj,
)
    _validate_csmc(model, csmc, observations, ref_traj)
    pf = csmc.pf
    K = length(observations)
    ref_state = _make_ref_state(ref_traj)

    # Backward predictive likelihoods (only non-nothing for RBPF)
    back_liks = _compute_backward_likelihoods(
        rng, model, _refreshment_filter(pf), observations, ref_state, csmc.refreshment
    )

    init_state = initialise(rng, model.prior, pf; ref_state)

    # Perform one CSMC-AS step on the current state
    function _csmc_as_step(state, t)
        rs = _step_resampler(rng, model, pf, t, state, observations[t])
        if !will_resample(rs, state)
            # The trigger depends only on the previous cloud, which is held fixed during
            # this ancestor update. Choosing the identity update on this branch is valid
            # also for ESS-dependent schedules, not only predetermined schedules.
            # Identity resampling carries the filtering weights. Retaining the reference
            # ancestor is a valid partial PGAS update; replacing it while retaining the
            # identity resampling weights is not the multinomial PGAS construction.
            state = preserve_sample(state)
        elseif isnothing(ref_state)
            state = resample(rng, rs, state)
        else
            ref_as = _build_ancestor_ref(ref_state, back_liks, t)
            as_weights = map(state.particles) do particle
                ancestor_weight(particle, model.dyn, pf, t, ref_as)
            end
            ancestor_idx = StatsBase.sample(rng, StatsBase.Weights(softmax(as_weights)))
            # Draw the WHOLE conditional law given the newly selected ancestor. For
            # dependent schemes, sampling conditional on ancestor 1 and overwriting it
            # afterwards gives the wrong distribution for all the other offspring.
            state = resample(rng, rs, state; ref_state, ref_idx=ancestor_idx)
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
    return _make_ref_state(trajectory), ll
end

## BACKWARD SIMULATION HELPERS #############################################################

# Backward simulation recomputes each suffix likelihood using the selected outer path.
_bs_init_back_lik(rng, model, pf, observations, K, state_K, strategy) = nothing
function _bs_init_back_lik(
    rng, model::HierarchicalSSM, pf::RBPF, observations, K, state_K::RBState, strategy
)
    return _backward_start(
        _backward_predictor(pf, strategy), model, pf, K, observations[K], state_K.x
    )
end
_build_bs_ref(state, ::Nothing) = state
_build_bs_ref(state::RBState, back_lik::AbstractLikelihood) = RBState(state.x, back_lik)

function _bs_step_back_lik(rng, model, pf, t, ::Nothing, observations, prev, next, strategy)
    return nothing
end
function _bs_step_back_lik(
    rng,
    model::HierarchicalSSM,
    pf::RBPF,
    t,
    back_lik::AbstractLikelihood,
    observations,
    prev_state::RBState,
    next_state::RBState,
    strategy,
)
    bp = _backward_predictor(pf, strategy)
    d = _component(inner_dynamics(model, t + 1, prev_state.x, next_state.x))
    pred_lik = backward_predict(bp, back_lik, d)
    return _backward_observe(
        bp, pred_lik, inner_observation(model, t, prev_state.x), t, observations[t]
    )
end

function _csmc_sample(
    rng::AbstractRNG,
    model::StateSpaceModel,
    csmc::ConditionalSMC{<:Any,<:BackwardSimulation},
    observations,
    ref_traj,
)
    _validate_csmc(model, csmc, observations, ref_traj)
    pf = csmc.pf
    K = length(observations)
    N = num_particles(pf)
    ref_state = _make_ref_state(ref_traj)

    # Forward filtering pass: store full history in a DenseParticleContainer.
    init_state = initialise(rng, model.prior, pf; ref_state)
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

    back_lik = _bs_init_back_lik(
        rng,
        model,
        _refreshment_filter(pf),
        observations,
        K,
        sampled_state,
        csmc.refreshment,
    )

    xs = Vector{typeof(sampled_state)}(undef, K)
    xs[K] = sampled_state

    for t in (K - 1):-1:1
        ref_next = _build_bs_ref(xs[t + 1], back_lik)
        backward_ws = map(1:N) do i
            ancestor_weight(Particle(container, t, i), model.dyn, pf, t + 1, ref_next)
        end
        idx = StatsBase.sample(rng, StatsBase.Weights(softmax(backward_ws)))
        xs[t] = container.states[t][idx]

        back_lik = _bs_step_back_lik(
            rng,
            model,
            _refreshment_filter(pf),
            t,
            back_lik,
            observations,
            xs[t],
            xs[t + 1],
            csmc.refreshment,
        )
    end

    # Time 0: backward step from t=1 to initial particles.
    ref_at_1 = _build_bs_ref(xs[1], back_lik)
    backward_ws = map(init_state.particles) do particle
        ancestor_weight(particle, model.dyn, pf, 1, ref_at_1)
    end
    idx = StatsBase.sample(rng, StatsBase.Weights(softmax(backward_ws)))
    x0 = container.initial_states[idx]

    return _make_ref_state(ReferenceTrajectory(x0, xs)), ll
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
