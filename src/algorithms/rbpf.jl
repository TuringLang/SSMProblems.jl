import LinearAlgebra: I
import Distributions: logpdf
import LogExpFunctions: softmax
import StatsBase: Weights

export RBPF

struct RBPF{F<:FilteringAlgorithm} <: FilteringAlgorithm
    n_particles::Int
    inner_algo::F
end

function filter(
    rng::AbstractRNG,
    model::HierarchicalSSM,
    algo::RBPF,
    observations::AbstractVector,
    extras::AbstractVector,
)
    T = length(observations)
    N = algo.n_particles
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    # Containers
    outer_type, inner_type = eltype(outer_dyn), rb_eltype(inner_model)
    xs = Vector{outer_type}(undef, N)
    zs = Vector{inner_type}(undef, N)
    new_xs = Vector{outer_type}(undef, N)
    new_zs = Vector{inner_type}(undef, N)
    log_ws = fill(-log(N), N)
    new_log_ws = similar(log_ws)

    # Initialisation
    for i in 1:N
        xs[i] = simulate(rng, outer_dyn)
        zs[i] = initialise(inner_model, algo.inner_algo)
    end

    for t in 1:T
        y = observations[t]
        u = extras[t]

        # Resampling
        weights = Weights(softmax(log_ws))
        parent_idxs = sample(rng, 1:N, weights, N)
        log_ws .= -log(N)

        for i in 1:N
            j = parent_idxs[i]
            new_xs[i] = simulate(rng, outer_dyn, t, xs[j], u)

            new_extras = (prev_outer=xs[j], new_outer=new_xs[i])
            inner_u = isnothing(u) ? new_extras : (; u..., new_extras...)

            new_zs[i], ll = step(inner_model, algo.inner_algo, t, zs[j], y, inner_u)
            new_log_ws[i] = log_ws[j] + ll
        end

        xs .= new_xs
        zs .= new_zs
        log_ws .= new_log_ws
    end

    return xs, zs, log_ws
end
