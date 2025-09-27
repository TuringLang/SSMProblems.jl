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
    rng::AbstractRNG,
    prior::HierarchicalPrior,
    algo::RBPF;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    x = sample_prior(rng, prior.outer_prior, algo.pf; ref_state, kwargs...)
    z = initialise(rng, prior.inner_prior, algo.af; new_outer=x, kwargs...)
    # TODO (RB):  determine the correct type for the log_w field or use a NoWeight type
    return RBParticle(x, z, 0.0, 0)
end

function predict_particle(
    rng::AbstractRNG,
    dyn::HierarchicalDynamics,
    algo::RBPF,
    iter::Integer,
    particle::RBParticle,
    observation;
    ref_state,
    kwargs...,
)
    new_x, logw_inc = propogate(
        rng, dyn.outer_dyn, algo.pf, iter, particle.x, observation; ref_state, kwargs...
    )
    new_z = predict(
        rng,
        dyn.inner_dyn,
        algo.af,
        iter,
        particle.z,
        observation;
        prev_outer=particle.x,
        new_outer=new_x,
        kwargs...,
    )

    return RBParticle(new_x, new_z, particle.log_w + logw_inc, particle.ancestor)
end

function update_particle(
    obs::ObservationProcess,
    algo::RBPF,
    iter::Integer,
    particle::RBParticle,
    observation;
    kwargs...,
)
    new_z, log_increment = update(
        obs, algo.af, iter, particle.z, observation; new_outer=particle.x, kwargs...
    )
    return RBParticle(particle.x, new_z, particle.log_w + log_increment, particle.ancestor)
end
