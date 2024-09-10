import LinearAlgebra: I
import Distributions: logpdf
import LogExpFunctions: softmax, logsumexp
import StatsBase: Weights

export RBPF

struct RBPF{F<:FilteringAlgorithm} <: FilteringAlgorithm
    inner_algo::F
    n_particles::Int
    resample_threshold::Float64
end
RBPF(inner_algo::F, n_particles::Int) where {F} = RBPF(inner_algo, n_particles, 1.0)

# TODO: rewrite this in terms of predict/update. This gets quite messy with the extra
# arguments. 
function step(rng, model::HierarchicalSSM, algo::RBPF, t::Integer, state, obs, extra)
    xs, zs, log_ws = state

    N = algo.n_particles
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    for i in 1:N
        prev_x = xs[i]
        xs[i] = simulate(rng, outer_dyn, t, prev_x, extra)

        new_extra = (prev_outer=prev_x, new_outer=xs[i])
        inner_extra = isnothing(extra) ? new_extra : (; extra..., new_extra...)

        zs[i], inner_ll = step(inner_model, algo.inner_algo, t, zs[i], obs, inner_extra)
        log_ws[i] = log_ws[i] + inner_ll
    end

    # TODO: this is probably incorrect
    ll = logsumexp(log_ws) - log(N)
    return (xs, zs, log_ws), ll
end

function filter(
    rng::AbstractRNG,
    model::HierarchicalSSM,
    algo::RBPF,
    observations::AbstractVector,
    extra0,
    extras::AbstractVector,
)
    N = algo.n_particles
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    # Containers
    outer_type, inner_type = eltype(outer_dyn), rb_eltype(inner_model)
    xs = Vector{outer_type}(undef, N)
    zs = Vector{inner_type}(undef, N)
    log_ws = fill(-log(N), N)

    # Initialisation
    ll = 0.0
    for i in 1:N
        xs[i] = simulate(rng, outer_dyn, extra0)
        new_extra0 = (; new_outer=xs[i])
        inner_extra0 = isnothing(extra0) ? new_extra0 : (; extra0..., new_extra0...)
        zs[i] = initialise(inner_model, algo.inner_algo, inner_extra0)
    end

    # Predict-update loop
    for (i, obs) in enumerate(observations)
        # Optional resampling
        weights = Weights(softmax(log_ws))
        ess = 1 / sum(weights .^ 2)
        if ess < algo.resample_threshold * N
            idxs = sample(rng, 1:N, weights, N)
            xs .= xs[idxs]
            zs .= zs[idxs]
            log_ws .= fill(-log(N), N)
        end
        (xs, zs, log_ws), step_ll = step(
            rng, model, algo, i, (xs, zs, log_ws), obs, extras[i]
        )
        ll += step_ll
    end
    return (xs, zs, log_ws), ll
end

function filter(
    model::HierarchicalSSM, algo::RBPF, observations::AbstractVector, extras::AbstractVector
)
    return filter(default_rng(), model, algo, observations, extras)
end

struct BatchRBPF{F<:FilteringAlgorithm} <: FilteringAlgorithm
    inner_algo::F
    n_particles::Int
    resample_threshold::Float64
end
function BatchRBPF(inner_algo::F, n_particles::Int) where {F}
    return BatchRBPF(inner_algo, n_particles, 1.0)
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

function step(
    model::HierarchicalSSM, algo::FilteringAlgorithm, t::Integer, state, obs, extra
)
    xs, zs, log_ws = state
    N = algo.n_particles
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    new_xs = batch_simulate(outer_dyn, N, xs)
    new_extras = (prev_outer=xs, new_outer=new_xs)
    inner_extra = isnothing(extra) ? new_extras : (; extra..., new_extras)

    zs, inner_lls = step(inner_model, algo.inner_algo, t, zs, obs, inner_extra)

    log_ws += inner_lls

    ll = logsumexp(log_ws) - log(N)
    return (xs, zs, log_ws), ll
end

function filter(
    model::HierarchicalSSM,
    algo::BatchRBPF,
    observations::AbstractVector,
    extras::AbstractVector,
)
    N = algo.n_particles
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    # Initialisation
    ll = 0.0
    xs = batch_simulate(outer_dyn, N)
    zs = initialise(inner_model, algo.inner_algo)
    log_ws = CUDA.fill(convert(Float32, -log(N)), N)

    # Predict-update loop
    for (i, obs) in enumerate(observations)
        # Optional resampling
        weights = softmax(log_ws)
        ess = 1 / sum(weights .^ 2)
        if ess < algo.resample_threshold * N
            cdf = cumsum(weights)
            us = CUDA.rand(N)
            idxs = CuArray{Int32}(N)
            @cuda threads = 256 blocks = 4096 searchsorted!(cdf, us, idxs)
            xs .= xs[:, idxs]
            # TODO: generalise this by creating a `resample` function
            μs = zs.μs[:, idxs]
            Σs = zs.Σs[:, :, idxs]
            zs = (μs=μs, Σs=Σs)
            log_ws .= convert(Float32, -log(N))
        end
        (xs, zs, log_ws), step_ll = step(
            model, algo.inner_algo, i, (xs, zs, log_ws), obs, extras[i]
        )
        ll += step_ll
    end
    return (xs, zs, log_ws), ll
end
