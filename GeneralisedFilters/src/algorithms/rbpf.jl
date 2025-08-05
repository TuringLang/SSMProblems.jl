import LinearAlgebra: I
import Distributions: logpdf
import LogExpFunctions: softmax, logsumexp
import StatsBase: Weights

export RBPF

struct RBPF{F<:AbstractFilter,RS<:AbstractResampler} <: AbstractParticleFilter
    inner_algo::F
    N::Int
    resampler::RS
end

function RBPF(
    inner_algo::AbstractFilter,
    N::Integer;
    threshold::Real=1.0,
    resampler::AbstractResampler=Systematic(),
)
    return RBPF(inner_algo, N, ESSResampler(threshold, resampler))
end

function initialise(
    rng::AbstractRNG,
    model::HierarchicalSSM{T},
    algo::RBPF;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
) where {T}
    particles = map(1:(algo.N)) do i
        x = if !isnothing(ref_state) && i == 1
            ref_state[0]
        else
            SSMProblems.simulate(rng, model.outer_prior; kwargs...)
        end
        z = initialise(rng, model.inner_model, algo.inner_algo; new_outer=x, kwargs...)
        RaoBlackwellisedParticle(x, z)
    end

    return Particles(particles)
end

function predict(
    rng::AbstractRNG,
    model::HierarchicalSSM,
    algo::RBPF,
    iter::Integer,
    state,
    observation;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    state.particles = map(enumerate(state.particles)) do (i, particle)
        new_x = if !isnothing(ref_state) && i == 1
            ref_state[iter]
        else
            SSMProblems.simulate(rng, model.outer_dyn, iter, particle.x; kwargs...)
        end
        new_z = predict(
            rng,
            model.inner_model,
            algo.inner_algo,
            iter,
            particle.z,
            observation;
            prev_outer=particle.x,
            new_outer=new_x,
            kwargs...,
        )

        RaoBlackwellisedParticle(new_x, new_z)
    end

    return state
end

function update(
    model::HierarchicalSSM, algo::RBPF, iter::Integer, state, observation; kwargs...
)
    log_increments = map(enumerate(state.particles)) do (i, particle)
        state.particles[i].z, log_increment = update(
            model.inner_model,
            algo.inner_algo,
            iter,
            particle.z,
            observation;
            new_outer=particle.x,
            kwargs...,
        )
        log_increment
    end

    state = update_weights(state, log_increments)
    ll_increment = marginalise!(state)

    return state, ll_increment
end
