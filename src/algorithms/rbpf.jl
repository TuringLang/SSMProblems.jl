import LinearAlgebra: I
import Distributions: logpdf
import LogExpFunctions: softmax, logsumexp
import StatsBase: Weights

export RBPF

struct RBPF{F<:AbstractFilter} <: AbstractFilter
    inner_algo::F
    n_particles::Int
    resample_threshold::Float64
end
RBPF(inner_algo::F, n_particles::Int) where {F} = RBPF(inner_algo, n_particles, 1.0)

function initialise(rng::AbstractRNG, model::HierarchicalSSM, algo::RBPF; kwargs...)
    N = algo.n_particles
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    # Create containers
    outer_type, inner_type = eltype(outer_dyn), rb_eltype(inner_model)
    particles = Vector{RaoBlackwellisedContainer{outer_type,inner_type}}(undef, N)
    log_ws = fill(-log(N), N)

    # Initialise containers
    for i in 1:N
        x = simulate(rng, outer_dyn; kwargs...)
        z = initialise(inner_model, algo.inner_algo; new_outer=x, kwargs...)
        particles[i] = RaoBlackwellisedContainer(x, z)
    end

    return ParticleContainer(particles, log_ws)
end

function predict(
    rng::AbstractRNG, model::HierarchicalSSM, algo::RBPF, t::Integer, states; kwargs...
)
    # TODO: reintroduce resampling
    # states.proposed = resample(rng, algo.resampler, states.filtered)
    states.proposed = deepcopy(states.filtered)

    for i in 1:(algo.n_particles)
        prev_x = states.proposed.particles[i].x
        states.proposed[i].x = simulate(rng, model.outer_dyn, t, prev_x; kwargs...)

        prev_z = states.proposed.particles[i].z
        states.proposed.particles[i].z = predict(
            rng,
            model.inner_model,
            algo.inner_algo,
            t,
            prev_z;
            prev_outer=prev_x,
            new_outer=states.proposed.particles[i].x,
            kwargs...,
        )
    end

    return states
end

function update(model::HierarchicalSSM, algo::RBPF, t::Integer, states, obs; kwargs...)
    # TODO: this type should be more general
    inner_lls = Vector{Float64}(undef, algo.n_particles)
    for i in 1:(algo.n_particles)
        states.filtered.particles[i].z, inner_ll = update(
            model.inner_model,
            algo.inner_algo,
            t,
            states.proposed.particles[i].z,
            obs;
            states.proposed.particles[i].x,
            kwargs...,
        )
        inner_lls[i] = inner_ll

        states.filtered.particles[i].x = states.proposed.particles[i].x
    end

    states.filtered.log_weights = states.proposed.log_weights .+ inner_lls

    return states, logsumexp(inner_lls) - log(algo.n_particles)
end
