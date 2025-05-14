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
    log_increments = map(1:(algo.N)) do i
        state.particles[i].z, log_increment = update(
            model.inner_model,
            algo.inner_algo,
            iter,
            state.particles[i].z,
            observation;
            new_outer=state.particles[i].x,
            kwargs...,
        )
        log_increment
    end

    state = update_weights(state, log_increments)
    return state, logmeanexp(log_increments)
end

#################################
#### GPU-ACCELERATED VERSION ####
#################################

# struct BatchRBPF{F<:AbstractFilter,RS<:AbstractResampler} <: AbstractParticleFilter
#     inner_algo::F
#     N::Int
#     resampler::RS
# end
# function BatchRBPF(
#     inner_algo, n_particles; threshold::Real=1.0, resampler::AbstractResampler=Systematic()
# )
#     return BatchRBPF(inner_algo, n_particles, ESSResampler(threshold, resampler))
# end

# function searchsorted!(ws_cdf, us, idxs)
#     index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride = blockDim().x * gridDim().x
#     for i in index:stride:length(us)
#         # Binary search
#         left = 1
#         right = length(ws_cdf)
#         while left < right
#             mid = (left + right) ÷ 2
#             if ws_cdf[mid] < us[i]
#                 left = mid + 1
#             else
#                 right = mid
#             end
#         end
#         idxs[i] = left
#     end
# end

# function initialise(
#     rng::AbstractRNG,
#     model::HierarchicalSSM,
#     algo::BatchRBPF;
#     ref_state::Union{Nothing,AbstractVector}=nothing,
#     kwargs...,
# )
#     N = algo.N
#     outer_dyn, inner_model = model.outer_dyn, model.inner_model

#     xs = SSMProblems.batch_simulate(rng, outer_dyn, N; ref_state, kwargs...)

#     # Set reference trajectory
#     if ref_state !== nothing
#         xs[:, 1] = ref_state[0]
#     end

#     zs = initialise(rng, inner_model, algo.inner_algo; new_outer=xs, kwargs...)
#     # log_ws = CUDA.zeros(T, N)

#     return RaoBlackwellisedParticleDistribution(
#         BatchRaoBlackwellisedParticles(xs, zs)
#     )
# end

# function predict(
#     rng::AbstractRNG,
#     model::HierarchicalSSM,
#     algo::BatchRBPF,
#     iter::Integer,
#     state::RaoBlackwellisedParticleDistribution,
#     observation;
#     ref_state::Union{Nothing,AbstractVector}=nothing,
#     kwargs...,
# )
#     outer_dyn, inner_model = model.outer_dyn, model.inner_model

#     new_xs = SSMProblems.batch_simulate(
#         rng, outer_dyn, iter, state.particles.xs; ref_state, kwargs...
#     )

#     # Set reference trajectory
#     if ref_state !== nothing
#         new_xs[:, [1]] = ref_state[iter]
#     end

#     new_zs = predict(
#         rng,
#         inner_model,
#         algo.inner_algo,
#         iter,
#         state.particles.zs,
#         observation;
#         prev_outer=state.particles.xs,
#         new_outer=new_xs,
#         kwargs...,
#     )
#     state.particles = BatchRaoBlackwellisedParticles(new_xs, new_zs)

#     return state
# end

# function update(
#     model::HierarchicalSSM,
#     algo::BatchRBPF,
#     iter::Integer,
#     state::RaoBlackwellisedParticleDistribution,
#     obs;
#     kwargs...,
# )
#     new_zs, inner_lls = update(
#         model.inner_model,
#         algo.inner_algo,
#         iter,
#         state.particles.zs,
#         obs;
#         new_outer=state.particles.xs,
#         kwargs...,
#     )

#     state.log_weights += inner_lls
#     state.particles.zs = new_zs

#     return state, logsumexp(state.log_weights)
# end
