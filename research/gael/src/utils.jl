module Utils

using Statistics
using GeneralisedFilters
using SSMProblems
using Random
using LinearAlgebra

export rand_cov, ensure_posdef, estimate_particle_count

function rand_cov(rng::AbstractRNG, T::Type{<:Real}, d::Int)
    Σ = rand(rng, T, d, d)
    return Σ * Σ'
end

function ensure_posdef(cov::AbstractMatrix{T}) where {T}
    cov_sym = Symmetric((cov + cov') / 2)
    I_mat = Matrix{T}(I, size(cov_sym)...)
    jitter = eps(real(T))^(0.5)
    for _ in 1:6
        try
            cholesky(cov_sym)
            return cov_sym
        catch
            cov_sym = Symmetric(cov_sym + jitter * I_mat)
            jitter *= 10
        end
    end
    return cov_sym
end

"""
    estimate_particle_count(rng, model, observations, filter_constructor; [target_variance=1.0, initial_N=50, num_replicates=20])

Estimate the number of particles required to achieve a target log-likelihood variance.
Returns `(estimated_N, observed_variance)`.
"""
function estimate_particle_count(
    rng::AbstractRNG,
    model::StateSpaceModel,
    observations::AbstractVector,
    filter_constructor::Function;
    target_variance::Float64 = 1.0,
    initial_N::Int = 50,
    num_replicates::Int = 20
)
    algo = filter_constructor(initial_N)
    logliks = Vector{Float64}(undef, num_replicates)
    
    for i in 1:num_replicates
        _, ll = GeneralisedFilters.filter(rng, model, algo, observations)
        logliks[i] = ll
    end
    
    current_var = var(logliks)
    
    if current_var < 1e-12
        return initial_N, current_var
    end
    
    projected_N = initial_N * (current_var / target_variance)
    return ceil(Int, projected_N), current_var
end

end
