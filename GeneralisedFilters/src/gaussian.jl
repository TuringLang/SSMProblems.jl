export GaussianState

"""
    GaussianState(μ, Σ)

A multivariate Gaussian with mean `μ` and covariance `Σ`, retaining the supplied array
representations. Covariances are symmetric by convention; positive-definiteness is not
enforced at construction. Kalman algorithms normalise their computational states to full
matrix storage independently of model parameters. Supports `rand` and `logpdf` when the
covariance meets the requirements of those operations.
"""
struct GaussianState{TM<:AbstractVector,TS<:AbstractMatrix}
    μ::TM
    Σ::TS
end

# Computational Kalman states use full covariance storage, independently of the
# parameter representation. Scalar promotion must include the covariance: its
# entries may carry Dual values even when the mean is constant.
function _kalman_state(μ::SVector{N}, Σ::AbstractMatrix) where {N}
    T = promote_type(eltype(μ), eltype(Σ))
    return GaussianState(SVector{N,T}(μ), SMatrix{N,N,T}(Σ))
end
function _kalman_state(μ::AbstractVector, Σ::AbstractMatrix)
    T = promote_type(eltype(μ), eltype(Σ))
    return GaussianState(convert(Vector{T}, μ), convert(Matrix{T}, Σ))
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

export CovarianceFactor, SqrtGaussianState

"""
    CovarianceFactor(F)

Covariance represented by an explicit factor `Σ = F*F'`. The factor may be rectangular
or rank deficient. Square-root algorithms consume `F` directly, without reconstructing
and refactorising `Σ`. This does not add noise or repair the statistical model.
"""
struct CovarianceFactor{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    factor::M
end
Base.size(C::CovarianceFactor) = (size(C.factor, 1), size(C.factor, 1))
function Base.getindex(C::CovarianceFactor, i::Int, j::Int)
    return sum(C.factor[i, k] * conj(C.factor[j, k]) for k in axes(C.factor, 2))
end
Base.Matrix(C::CovarianceFactor) = Matrix(C.factor * C.factor')
_covariance_root(C::CovarianceFactor) = C.factor
_covariance_root(C::AbstractMatrix) = cholesky(Symmetric(C)).L

Statistics.mean(g::SqrtGaussianState) = g.μ
Statistics.cov(g::SqrtGaussianState) = symmetrise(g.U' * g.U)
Base.length(g::SqrtGaussianState) = length(g.μ)
Base.eltype(::Type{SqrtGaussianState{TM,TU}}) where {TM,TU} = eltype(TM)
Random.rand(rng::AbstractRNG, g::SqrtGaussianState) = g.μ + g.U' * _randn_like(rng, g.μ)
function Random.rand(
    rng::AbstractRNG, g::GaussianState{<:AbstractVector,<:CovarianceFactor}
)
    F = g.Σ.factor
    return g.μ + F * _factor_randn(rng, F)
end

_factor_randn(rng, F::AbstractMatrix) = randn(rng, eltype(F), size(F, 2))
_factor_randn(rng, F::StaticMatrix{N,M,T}) where {N,M,T} = SVector{M,T}(randn(rng, T, M))
