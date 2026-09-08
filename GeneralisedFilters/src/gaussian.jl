export GaussianState

"""
    GaussianState(μ, Σ)

A multivariate Gaussian with mean `μ` and covariance `Σ`. `Σ` is stored as a plain matrix,
symmetric by convention; positive-definiteness is not enforced. Serves both as the Kalman
filtering state and as a lightweight distribution supporting `rand` and `logpdf`.
"""
struct GaussianState{TM<:AbstractVector,TS<:AbstractMatrix}
    μ::TM
    Σ::TS
end

Statistics.mean(g::GaussianState) = g.μ
Distributions.mode(g::GaussianState) = g.μ
Statistics.cov(g::GaussianState) = g.Σ
Base.length(g::GaussianState) = length(g.μ)
Base.eltype(::Type{GaussianState{TM,TS}}) where {TM,TS} = eltype(TM)

symmetrise(A::AbstractMatrix) = (A + A') / 2

# Standard normal draw preserving the container type of the mean.
_randn_like(rng::AbstractRNG, ::SVector{D,T}) where {D,T} = @SVector randn(rng, T, D)
_randn_like(rng::AbstractRNG, μ::AbstractVector{T}) where {T} = randn(rng, T, length(μ))

function Random.rand(rng::AbstractRNG, g::GaussianState)
    return g.μ + cholesky(Symmetric(g.Σ)).L * _randn_like(rng, g.μ)
end

function Distributions.logpdf(g::GaussianState, x::AbstractVector)
    C = cholesky(Symmetric(g.Σ))
    z = x - g.μ
    return -(length(x) * log(2π) + logdet(C) + dot(z, C \ z)) / 2
end

# Boundary interop: wrap in a PDMat exactly once, only when explicitly requested.
function Distributions.MvNormal(g::GaussianState)
    return MvNormal(Vector(g.μ), Matrix(Symmetric(g.Σ)))
end

function Base.isapprox(a::GaussianState, b::GaussianState; kwargs...)
    return isapprox(a.μ, b.μ; kwargs...) && isapprox(a.Σ, b.Σ; kwargs...)
end

"""
    SqrtGaussianState(μ, U)

A Gaussian in square-root form, where the covariance is `Σ = U'U` with `U` upper triangular.
Used by the square-root Kalman filter to propagate the covariance factor directly.
"""
struct SqrtGaussianState{TM<:AbstractVector,TU<:UpperTriangular}
    μ::TM
    U::TU
end

GaussianState(g::SqrtGaussianState) = GaussianState(g.μ, symmetrise(g.U' * g.U))
SqrtGaussianState(g::GaussianState) = SqrtGaussianState(g.μ, cholesky(Symmetric(g.Σ)).U)
