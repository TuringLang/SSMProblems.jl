import LinearAlgebra: I
import Distributions: logpdf
import LogExpFunctions: softmax, logsumexp
import StatsBase: Weights

export RBPF, RBState

"""
    RBPF(particle_filter, analytical_filter)

Rao–Blackwellised particle filter for hierarchical models. Particle proposals receive
an `RBState` and return the outer sample; the analytical filter then resolves and updates
the conditional inner model. Reference trajectories contain outer samples only.
"""
struct RBPF{PFT<:AbstractParticleFilter,AFT<:AbstractFilter} <: AbstractParticleFilter
    pf::PFT
    af::AFT
end

num_particles(algo::RBPF) = num_particles(algo.pf)
resampler(algo::RBPF) = resampler(algo.pf)
execution(algo::RBPF) = execution(algo.pf)

function initialise_particle(
    rng::AbstractRNG, prior::HierarchicalPrior, algo::RBPF, ref_state
)
    x = sample_prior(rng, prior.outer, algo.pf, ref_state)
    z = initialise(rng, _component(inner_prior(prior, x)), algo.af)
    return Particle(RBState(x, z), 0)
end

function predict_particle(
    rng::AbstractRNG,
    dyn::HierarchicalDynamics,
    algo::RBPF,
    iter::Integer,
    particle::Particle{<:RBState},
    observation,
    ref_state,
)
    state, logw_inc = _predict_rb_state(
        rng, dyn, algo, iter, particle.state, observation, ref_state
    )
    return Particle(state, add_logweight(log_weight(particle), logw_inc), particle.ancestor)
end

# The RBPF recipe acts on an RBState whose fields may be scalar beliefs/samples
# or batched collections. Propagation and the analytical filter select execution
# through their existing dispatch, keeping the conditional model logic shared.
function _predict_rb_state(
    rng::AbstractRNG,
    dyn::HierarchicalDynamics,
    algo::RBPF,
    iter::Integer,
    state::RBState,
    observation,
    ref_state,
)
    new_x, logw_inc = propagate(
        rng, dyn.outer, algo.pf, iter, state, observation, ref_state
    )
    new_z = predict(
        rng,
        _component(inner_dynamics(dyn, iter, state.x, new_x)),
        algo.af,
        iter,
        state.z,
        observation,
    )
    return RBState(new_x, new_z), logw_inc
end

function update_particle(
    obs::ObservationProcess,
    algo::RBPF,
    iter::Integer,
    particle::Particle{<:RBState},
    observation,
)
    state, log_increment = _update_rb_state(obs, algo, iter, particle.state, observation)
    return Particle(
        state, add_logweight(log_weight(particle), log_increment), particle.ancestor
    )
end

function _update_rb_state(
    obs::ObservationProcess, algo::RBPF, iter::Integer, state::RBState, observation
)
    new_z, log_increment = update(
        _component(inner_observation(obs, iter, state.x)),
        algo.af,
        iter,
        state.z,
        observation,
    )
    return RBState(state.x, new_z), log_increment
end

function predictive_state(
    rng::AbstractRNG,
    dyn::HierarchicalDynamics,
    weight_strategy::RepresentativeStateLookAhead,
    rbpf::RBPF,
    iter::Integer,
    state::RBState,
)
    x_star = predictive_statistic(rng, weight_strategy.pp, dyn.outer, iter, state.x)
    z_star = predict(
        rng,
        _component(inner_dynamics(dyn, iter, state.x, x_star)),
        rbpf.af,
        iter,
        state.z,
        nothing,
    )
    return RBState(x_star, z_star)
end

function predictive_loglik(
    obs::ObservationProcess, algo::RBPF, iter::Integer, state::RBState, observation
)
    _, log_increment = update(
        _component(inner_observation(obs, iter, state.x)),
        algo.af,
        iter,
        state.z,
        observation,
    )
    return log_increment
end
