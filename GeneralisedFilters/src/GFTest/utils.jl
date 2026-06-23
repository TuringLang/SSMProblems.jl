"""
    make_pd(M)

Project a square matrix to the nearest positive-definite matrix by symmetrising and
clipping negative eigenvalues to `1e-8`. Returns a `PDMat`.
"""
function make_pd(M)
    M_sym = (M + M') / 2
    λ, V = eigen(M_sym)
    λ_clipped = max.(λ, 1e-8)
    return PDMat(Symmetric(V * Diagonal(λ_clipped) * V'))
end

function rand_cov(
    rng::AbstractRNG, T::Type{<:Real}, d::Int; scale=T(1.0), var_range=(T(0.8), T(1.2))
)
    A = rand(rng, T, d, d)
    Σ = A * A'

    # Scale the diagonal to get a correlation matrix
    D_inv_sqrt = Diagonal(1 ./ sqrt.(diag(Σ)))
    Σ = D_inv_sqrt * Σ * D_inv_sqrt

    # Generate variances for each dimension
    variances = rand(rng, T, d) .* (var_range[2] - var_range[1]) .+ var_range[1]
    variances *= scale

    # Scale the by these variances
    D_sqrt = Diagonal(sqrt.(variances))
    Σ = D_sqrt * Σ * D_sqrt

    # Ensure the covariance matrix symmetric
    Σ = (Σ + Σ') / 2

    return Σ
end
