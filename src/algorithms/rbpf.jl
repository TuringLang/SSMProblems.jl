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
        xs[i] = simulate(rng, outer_dyn)
        zs[i] = initialise(inner_model, algo.inner_algo)
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
