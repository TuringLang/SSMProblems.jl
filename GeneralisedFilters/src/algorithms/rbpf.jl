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

# Population adapters exchange an RBState of field collections. These are borrowed
# inputs, not writable projections into an array of particles. Assembly builds a
# new population and borrows the supplied leaves (no implicit deep copy).
function _rb_population_fields(particles::AbstractVector{<:Particle{<:RBState}})
    return (;
        state=RBState(map(p -> p.state.x, particles), map(p -> p.state.z, particles)),
        log_w=map(log_weight, particles),
        ancestor=map(p -> p.ancestor, particles),
    )
end

function _check_rb_population_fields(particles, fields)
    n = length(particles)
    all(
        length(c) == n for
        c in (fields.state.x, fields.state.z, fields.log_w, fields.ancestor)
    ) || throw(DimensionMismatch("RBPF population field lengths differ"))
    return nothing
end

function _assemble_rb_population(particles, fields::NamedTuple)
    _check_rb_population_fields(particles, fields)
    return map(eachindex(particles)) do i
        return Particle(
            RBState(fields.state.x[i], fields.state.z[i]),
            fields.log_w[i],
            fields.ancestor[i],
        )
    end
end

# A CPU traversal already constructs its output particles. Preserve that result
# instead of splitting and rebuilding it merely to cross the population boundary.
function _assemble_rb_population(particles, result::AbstractVector{<:Particle{<:RBState}})
    length(result) == length(particles) ||
        throw(DimensionMismatch("RBPF population lengths differ"))
    return result
end

# Both representations use this orchestration. Execution adapters retain the CPU
# scalar hooks/RNG traversal or evaluate the same RBPF recipe on whole batches.
function _predict_particles(
    rng::AbstractRNG,
    dyn::HierarchicalDynamics,
    algo::RBPF{<:BootstrapFilter,<:KalmanFilter},
    t::Integer,
    particles::AbstractVector{<:Particle{<:RBState}},
    observation,
    ref_state,
)
    reference = _reference_state(ref_state, t)
    fields = _predict_rb_population(rng, dyn, algo, t, particles, observation, reference)
    return _assemble_rb_population(particles, fields)
end

function _predict_rb_population(rng, dyn, algo, t, particles, observation, reference)
    predicted = _population_map(execution(algo), rng, length(particles)) do rng, i
        return predict_particle(
            rng, dyn, algo, t, particles[i], observation, i == 1 ? reference : nothing
        )
    end
    return predicted
end

function _update_particles(
    obs::HierarchicalObservation,
    algo::RBPF{<:BootstrapFilter,<:KalmanFilter},
    t::Integer,
    particles::AbstractVector{<:Particle{<:RBState}},
    observation,
)
    fields = _update_rb_population(obs, algo, t, particles, observation)
    return _assemble_rb_population(particles, fields)
end

function _update_rb_population(obs, algo, t, particles, observation)
    filtered = _population_map(execution(algo), length(particles)) do i
        return update_particle(obs, algo, t, particles[i], observation)
    end
    return filtered
end

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
