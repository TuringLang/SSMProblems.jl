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
    particles = map(1:(algo.N)) do i
        x = if !isnothing(ref_state) && i == 1
            ref_state[0]
        else
            SSMProblems.simulate(rng, model.outer_dyn; kwargs...)
        end
        z = initialise(rng, model.inner_model, algo.inner_algo; new_outer=x, kwargs...)

        RaoBlackwellisedParticle(x, z)
    end
    log_ws = zeros(T, algo.N)

    return ParticleDistribution(particles, log_ws)
end

function predict(
    rng::AbstractRNG,
    model::HierarchicalSSM,
    algo::RBPF,
    t::Integer,
    state;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    # state.particles = map(
    #     i -> predict_particle(
    #         rng, i, model, algo, t, state.particles[i]; ref_state, kwargs...
    #     ),
    #     1:(algo.N),
    # )

    # return state
    state.particles = map(1:(algo.N)) do i
        new_x = if !isnothing(ref_state) && i == 1
            ref_state[t]
        else
            SSMProblems.simulate(rng, model.outer_dyn, t, state.particles[i].x; kwargs...)
        end
        new_z = predict(
            rng,
            model.inner_model,
            algo.inner_algo,
            t,
            state.particles[i].z;
            prev_outer=state.particles[i].x,
            new_outer=new_x,
            kwargs...,
        )

        RaoBlackwellisedParticle(new_x, new_z)
    end

    return state
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

    xs = SSMProblems.batch_simulate(rng, outer_dyn, N; ref_state, kwargs...)

    # Set reference trajectory
    if ref_state !== nothing
        xs[:, 1] = ref_state[0]
    end

    zs = initialise(rng, inner_model, algo.inner_algo; new_outer=xs, kwargs...)
    log_ws = CUDA.zeros(T, N)

    return RaoBlackwellisedParticleDistribution(
        BatchRaoBlackwellisedParticles(xs, zs), log_ws
    )
end

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

    new_xs = SSMProblems.batch_simulate(
        rng, outer_dyn, step, state.particles.xs; ref_state, kwargs...
    )

    # Set reference trajectory
    if ref_state !== nothing
        new_xs[:, [1]] = ref_state[step]
    end

    new_zs = predict(
        rng,
        inner_model,
        filter.inner_algo,
        step,
        state.particles.zs;
        prev_outer=state.particles.xs,
        new_outer=new_xs,
        kwargs...,
    )
    state.particles = BatchRaoBlackwellisedParticles(new_xs, new_zs)

    return state
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
