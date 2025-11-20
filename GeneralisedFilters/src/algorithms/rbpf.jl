import LinearAlgebra: I
import Distributions: logpdf
import LogExpFunctions: softmax, logsumexp
import StatsBase: Weights

export RBPF

struct RBPF{PFT<:AbstractParticleFilter,AFT<:AbstractFilter} <: AbstractParticleFilter
    pf::PFT
    af::AFT
end

num_particles(algo::RBPF) = num_particles(algo.pf)
resampler(algo::RBPF) = resampler(algo.pf)

function initialise_particle(
    rng::AbstractRNG, prior::HierarchicalPrior, algo::RBPF, ref_state; kwargs...
)
    N = num_particles(algo)
    x = sample_prior(rng, prior.outer_prior, algo.pf, ref_state; kwargs...)
    z = initialise(rng, prior.inner_prior, algo.af; new_outer=x, kwargs...)
    # TODO (RB):  determine the correct type for the log_w field or use a NoWeight type
    # return Particle(RBState(x, z), -log(N), 0)
    return Particle(RBState(x, z), 0)
end

function predict_particle(
    rng::AbstractRNG,
    dyn::HierarchicalDynamics,
    algo::RBPF,
    iter::Integer,
    particle::AbstractParticle{<:RBState},
    observation,
    ref_state;
    kwargs...,
)
    # TODO: really we should be conditioning on the current RB state to allow for optimal proposals
    new_x, logw_inc = propogate(
        rng,
        dyn.outer_dyn,
        algo.pf,
        iter,
        particle.state.x,
        observation,
        ref_state;
        kwargs...,
    )
    new_z = predict(
        rng,
        dyn.inner_dyn,
        algo.af,
        iter,
        particle.state.z,
        observation;
        prev_outer=particle.state.x,
        new_outer=new_x,
        kwargs...,
    )

    return Particle(
        RBState(new_x, new_z), log_weight(particle) + logw_inc, particle.ancestor
    )
end

function update_particle(
    obs::ObservationProcess,
    algo::RBPF,
    iter::Integer,
    particle::AbstractParticle{<:RBState},
    observation;
    kwargs...,
)
    new_z, log_increment = update(
        obs,
        algo.af,
        iter,
        particle.state.z,
        observation;
        new_outer=particle.state.x,
        kwargs...,
    )
    return Particle(
        RBState(particle.state.x, new_z),
        log_weight(particle) + log_increment,
        particle.ancestor,
    )
end

function predictive_state(
    rng::AbstractRNG,
    dyn::HierarchicalDynamics,
    apf::AuxiliaryParticleFilter{<:RBPF},
    iter::Integer,
    particle::AbstractParticle{<:RBState};
    kwargs...,
)
    rbpf = apf.pf
    x_star = predictive_statistic(
        rng, apf.pp, dyn.outer_dyn, iter, particle.state.x; kwargs...
    )
    z_star = predict(
        rng,
        dyn.inner_dyn,
        rbpf.af,
        iter,
        particle.state.z,
        nothing;  # no observation available — maybe we should pass this in
        prev_outer=particle.state.x,
        new_outer=x_star,
        kwargs...,
    )
    return Particle(RBState(x_star, z_star), particle.log_w, particle.ancestor)
end

function predictive_loglik(
    obs::ObservationProcess,
    algo::RBPF,
    iter::Integer,
    p_star::AbstractParticle{<:RBState},
    observation;
    kwargs...,
)
    _, log_increment = update(
        obs, algo.af, iter, p_star.state.z, observation; new_outer=p_star.state.x, kwargs...
    )
    return log_increment
end
