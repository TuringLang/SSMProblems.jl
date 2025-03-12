import LinearAlgebra: I
import Distributions: logpdf
import LogExpFunctions: softmax, logsumexp
import StatsBase: Weights

export RBPF, BatchRBPF

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
    particles = map(
        x -> RaoBlackwellisedParticle(
            simulate(rng, model.outer_dyn; kwargs...),
            initialise(model.inner_model, algo.inner_algo; new_outer=x, kwargs...),
        ),
        1:(algo.N),
    )
    log_ws = zeros(T, algo.N)

    return update_ref!(ParticleDistribution(particles, log_ws), ref_state)
end

function predict(
    rng::AbstractRNG,
    model::HierarchicalSSM,
    algo::RBPF,
    t::Integer,
    filtered;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    new_particles = map(
        x -> marginal_predict(rng, model, algo, t, x; kwargs...), filtered.particles
    )
    # Don't need to deep copy weights as filtered will be overwritten in the update step
    proposed = ParticleDistribution(new_particles, filtered.log_weights)

    return update_ref!(proposed, ref_state, t)
end

function marginal_predict(
    rng::AbstractRNG, model::HierarchicalSSM, algo::RBPF, t::Integer, state; kwargs...
)
    proposed_x = simulate(rng, model.outer_dyn, t, state.x; kwargs...)
    proposed_z = predict(
        rng,
        model.inner_model,
        algo.inner_algo,
        t,
        state.z;
        prev_outer=state.x,
        new_outer=proposed_x,
        kwargs...,
    )

    return RaoBlackwellisedParticle(proposed_x, proposed_z)
end

function update(
    model::HierarchicalSSM{T}, algo::RBPF, t::Integer, state, obs; kwargs...
) where {T}
    old_ll = logsumexp(state.log_weights)

    for i in 1:(algo.N)
        state.particles[i].z, log_increments = update(
            model.inner_model,
            algo.inner_algo,
            t,
            state.particles[i].z,
            obs;
            new_outer=state.particles[i].x,
            kwargs...,
        )
        state.log_weights[i] += log_increments
    end

    ll_increment = logsumexp(state.log_weights) - old_ll

    return state, ll_increment
end

function marginal_update(
    model::HierarchicalSSM, algo::RBPF, t::Integer, state, obs; kwargs...
)
    filtered_z, log_increment = update(
        model.inner_model, algo.inner_algo, t, state.z, obs; new_outer=state.x, kwargs...
    )

    return RaoBlackwellisedParticle(state.x, filtered_z), log_increment
end

#################################
#### GPU-ACCELERATED VERSION ####
#################################

struct BatchRBPF{F<:AbstractFilter,RS<:AbstractResampler} <: AbstractParticleFilter
    inner_algo::F
    N::Int
    resampler::RS
end
function BatchRBPF(
    inner_algo, n_particles; threshold::Real=1.0, resampler::AbstractResampler=Systematic()
)
    return BatchRBPF(inner_algo, n_particles, ESSResampler(threshold, resampler))
end

function searchsorted!(ws_cdf, us, idxs)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i in index:stride:length(us)
        # Binary search
        left = 1
        right = length(ws_cdf)
        while left < right
            mid = (left + right) ÷ 2
            if ws_cdf[mid] < us[i]
                left = mid + 1
            else
                right = mid
            end
        end
        idxs[i] = left
    end
end

function initialise(
    rng::AbstractRNG,
    model::HierarchicalSSM{T},
    algo::BatchRBPF;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
) where {T}
    N = algo.N
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    xs = SSMProblems.batch_simulate(rng, outer_dyn, N; kwargs...)
    zs = initialise(inner_model, algo.inner_algo; new_outer=xs, kwargs...)
    log_ws = CUDA.zeros(T, N)

    return update_ref!(
        RaoBlackwellisedParticleDistribution(
            BatchRaoBlackwellisedParticles(xs, zs), log_ws
        ),
        ref_state,
    )
end

# TODO: use RNG
function predict(
    rng::AbstractRNG,
    model::HierarchicalSSM,
    filter::BatchRBPF,
    step::Integer,
    state::RaoBlackwellisedParticleDistribution;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    new_xs = SSMProblems.batch_simulate(rng, outer_dyn, step, state.particles.xs; kwargs...)
    new_zs = predict(
        inner_model,
        filter.inner_algo,
        step,
        state.particles.zs;
        prev_outer=state.particles.xs,
        new_outer=new_xs,
        kwargs...,
    )
    state.particles = BatchRaoBlackwellisedParticles(new_xs, new_zs)

    return update_ref!(state, ref_state, step)
end

function update(
    model::HierarchicalSSM,
    filter::BatchRBPF,
    step::Integer,
    state::RaoBlackwellisedParticleDistribution,
    obs;
    kwargs...,
)
    old_ll = logsumexp(state.log_weights)

    new_zs, inner_lls = update(
        model.inner_model,
        filter.inner_algo,
        step,
        state.particles.zs,
        obs;
        new_outer=state.particles.xs,
        kwargs...,
    )

    state.log_weights += inner_lls
    state.particles.zs = new_zs

    step_ll = logsumexp(state.log_weights) - old_ll
    return state, step_ll
end
