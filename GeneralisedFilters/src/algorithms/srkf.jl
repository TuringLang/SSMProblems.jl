using StaticArrays: SOneTo, SUnitRange
using LinearAlgebra: qr, UpperTriangular, Diagonal, cholesky, diag, dot

export SRKalmanFilter, SRKF

"""
    SRKalmanFilter()

Square-root Kalman filter for linear-Gaussian state-space models. Propagates the Cholesky
factor of the covariance directly via QR factorisation, keeping the covariance positive
semi-definite by construction (no repair is required).

The covariance is represented as `Σ = U'U` with `U` upper triangular; the filtering state is
a [`SqrtGaussianState`](@ref). Explicit [`CovarianceFactor`](@ref)s may supply
rank-deficient prior and process-noise roots. Plain covariance arrays are Cholesky-factorized
and must be positive definite. Observation noise must be positive definite.

Likelihood differentiation is supported on nondegenerate QR branches. Derivatives through
rank changes or zero QR pivots are not supported; singular filtering support does not imply
smooth derivatives of a chosen factor representation.
"""
struct SRKalmanFilter <: AbstractFilter end
SRKF() = SRKalmanFilter()

"""
    marginal_loglikelihood(model, ::SRKalmanFilter, ys)

Evaluate the deterministic square-root Kalman likelihood through the ordinary filter.
"""
function marginal_loglikelihood(
    model::StateSpaceModel, af::SRKalmanFilter, ys::AbstractVector
)
    return last(filter(model, af, ys))
end

function initialise(
    ::AbstractRNG, prior::GaussianPrior, ::SRKalmanFilter; ref_state=nothing
)
    return SqrtGaussianState(prior.μ0, _upper_covariance_root(prior.Σ0))
end

function predict(
    ::AbstractRNG,
    dyn,
    ::SRKalmanFilter,
    t::Integer,
    state::SqrtGaussianState,
    y;
    ref_state=nothing,
)
    return srkf_predict(state, _component(resolve(dyn, (; t))))
end

function update(obs, ::SRKalmanFilter, t::Integer, state::SqrtGaussianState, y)
    return srkf_update(state, _component(resolve(obs, (; t))), y)
end

"""
    _correct_cholesky_sign(R)

Flip rows of an upper triangular matrix so its diagonal is positive, as required for use as a
Cholesky factor (QR factorisation may return negative diagonal entries).
"""
_qr_upper(M) = qr(M).R

_correct_cholesky_sign(R) = Diagonal(map(x -> x < zero(x) ? -one(x) : one(x), diag(R))) * R

_root_padding(F::AbstractMatrix) = zeros(eltype(F), size(F, 1), size(F, 1))
_root_padding(F::StaticMatrix{N,M,T}) where {N,M,T} = zero(SMatrix{N,N,T})

_upper_covariance_root(C::AbstractMatrix) = cholesky(Symmetric(C)).U
function _upper_covariance_root(C::CovarianceFactor)
    F = C.factor
    # Padding supports a rectangular, even zero-rank, initial covariance factor.
    R = _qr_upper(vcat(F', _root_padding(F)))
    return UpperTriangular(_correct_cholesky_sign(R))
end

function srkf_predict(state::SqrtGaussianState, d::LinearGaussianDynamics)
    μ, U = state.μ, state.U
    U_Q = _covariance_root(d.Q)'
    μ̂ = d.A * μ + d.b
    Û = _srkf_predict_covariance(U, d.A, U_Q)
    return SqrtGaussianState(μ̂, UpperTriangular(Û))
end

function _srkf_predict_covariance(U, A, U_Q)
    M = vcat(U_Q, U * A')
    R = _qr_upper(M)
    return _correct_cholesky_sign(R)
end

function srkf_update(state::SqrtGaussianState, o::LinearGaussianObservation, y)
    μ, U = state.μ, state.U
    U_R = _upper_covariance_root(o.R)
    μ̂, Û, ll = _srkf_update_covariance(μ, U, y, o.H, o.c, U_R)
    return SqrtGaussianState(μ̂, UpperTriangular(Û)), ll
end

function _srkf_update_covariance(μ, U, y, H, c, U_R)
    Dy = size(H, 1)
    Dx = size(H, 2)

    M = _srkf_form_update_matrix(U, H, U_R)
    R = _qr_upper(M)
    R = _correct_cholesky_sign(R)

    U_S, PHt, Û = _srkf_extract_update_components(R, H)

    z = y - H * μ - c
    w = U_S' \ z
    μ̂ = μ + PHt * w

    ll = _srkf_loglikelihood(w, U_S, Dy)

    return μ̂, Û, ll
end

function _srkf_form_update_matrix(U, H, U_R)
    Dy = size(H, 1)
    Dx = size(H, 2)
    T = promote_type(eltype(U), eltype(H), eltype(U_R))

    M = zeros(T, Dy + Dx, Dy + Dx)
    M[1:Dy, 1:Dy] = U_R
    M[(Dy + 1):end, 1:Dy] = U * H'
    M[(Dy + 1):end, (Dy + 1):end] = U
    return M
end

function _srkf_extract_update_components(R, H)
    Dy, Dx = size(H)
    U_S = R[1:Dy, 1:Dy]
    PHt = R[1:Dy, (Dy + 1):(Dy + Dx)]'
    Û = R[(Dy + 1):(Dy + Dx), (Dy + 1):(Dy + Dx)]
    return U_S, PHt, Û
end

function _srkf_loglikelihood(w, U_S, Dy)
    return -0.5 * (dot(w, w) + 2 * sum(log.(diag(U_S))) + Dy * log(2π))
end

# StaticArrays specialisations for type stability

function _srkf_predict_covariance(
    U::UpperTriangular{TU,<:SMatrix{Dx,Dx,TU}},
    A::SMatrix{Dx,Dx,TA},
    U_Q::UpperTriangular{TQ,<:SMatrix{Dx,Dx,TQ}},
) where {Dx,TU,TA,TQ}
    M = vcat(parent(U_Q), parent(U) * A')
    R = _qr_upper(M)
    return _correct_cholesky_sign(R)
end

function _srkf_form_update_matrix(
    U::UpperTriangular{TU,<:SMatrix{Dx,Dx,TU}},
    H::SMatrix{Dy,Dx,TH},
    U_R::UpperTriangular{TR,<:SMatrix{Dy,Dy,TR}},
) where {Dx,Dy,TU,TH,TR}
    T = promote_type(TU, TH, TR)
    top = hcat(parent(U_R), @SMatrix zeros(T, Dy, Dx))
    bottom = hcat(parent(U) * H', parent(U))
    return vcat(top, bottom)
end

function _srkf_extract_update_components(
    R::SMatrix{N,N,T}, ::StaticMatrix{Dy,Dx}
) where {N,T,Dy,Dx}
    U_S = R[SOneTo(Dy), SOneTo(Dy)]
    PHt = R[SOneTo(Dy), SUnitRange(Dy + 1, Dy + Dx)]'
    Û = R[SUnitRange(Dy + 1, Dy + Dx), SUnitRange(Dy + 1, Dy + Dx)]
    return U_S, PHt, Û
end
