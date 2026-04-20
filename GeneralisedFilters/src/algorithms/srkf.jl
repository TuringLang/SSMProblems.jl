using StaticArrays: SOneTo, SUnitRange
using LinearAlgebra: qr, UpperTriangular, Diagonal, Cholesky, cholesky, diag, dot

export SRKalmanFilter, SRKF

"""
    SRKalmanFilter(; jitter=nothing)

Square-root Kalman filter for linear Gaussian state space models.

Uses QR factorization to propagate the Cholesky factor of the covariance matrix
directly, avoiding the numerical instabilities associated with forming and
subtracting full covariance matrices.

# Fields
- `jitter::Union{Nothing, Real}`: Optional value added to the covariance matrix after the
  update step to improve numerical stability. If `nothing`, no jitter is applied.

# Algorithm
The SRKF represents the covariance as `Σ = U' * U` where `U` is upper triangular.

**Predict Step**: Given filtered state:
1. Form matrix `A = [[√Q], [A*U']]`
2. QR factorize to obtain `U_new` (predicted square-root covariance)

**Update Step**: Given predicted state and observation `y`:
1. Form matrix `A = [[√R, H*U'], [0, U']]`
2. QR factorize: `Q, B = qr(A)` where `B` is upper triangular
3. Extract `U_new` (posterior square-root covariance) from bottom-right block
4. Compute Kalman gain from the factorization components

See also: [`KalmanFilter`](@ref)
"""
struct SRKalmanFilter{T<:Union{Nothing,Real}} <: AbstractFilter
    jitter::T
end
SRKalmanFilter(; jitter=nothing) = SRKalmanFilter(jitter)

SRKF() = SRKalmanFilter()

function initialise(
    rng::AbstractRNG, prior::GaussianPrior, filter::SRKalmanFilter; kwargs...
)
    μ0, Σ0 = calc_initial(prior; kwargs...)
    return MvNormal(μ0, Σ0)
end

function predict(
    rng::AbstractRNG,
    dyn::LinearGaussianLatentDynamics,
    algo::SRKalmanFilter,
    iter::Integer,
    state::MvNormal,
    observation=nothing;
    kwargs...,
)
    dyn_params = calc_params(dyn, iter; kwargs...)
    return srkf_predict(state, dyn_params)
end

function update(
    obs::LinearGaussianObservationProcess,
    algo::SRKalmanFilter,
    iter::Integer,
    state::MvNormal,
    observation::AbstractVector;
    kwargs...,
)
    obs_params = calc_params(obs, iter; kwargs...)
    state, ll = srkf_update(state, obs_params, observation, algo.jitter)
    return state, ll
end

"""
    _correct_cholesky_sign(R)

Ensure the diagonal of an upper triangular matrix is positive.

QR factorization produces an upper triangular R that may have negative diagonal elements. 
For use as a Cholesky factor, the diagonal must be positive. This function multiplies rows
by -1 as needed, in an StaticArray-compatible fashion.
"""
_correct_cholesky_sign(R) = Diagonal(sign.(diag(R))) * R

"""
    srkf_predict(state, dyn_params)

Perform the square-root Kalman filter predict step.

Given the filtered state and dynamics parameters `(A, b, Q)`, compute the predicted
state using QR factorization.
"""
function srkf_predict(state::MvNormal, dyn_params)
    μ, Σ = params(state)
    A, b, Q = dyn_params

    U = cholesky(Σ).U
    U_Q = cholesky(Q).U

    μ̂ = A * μ + b
    Û = _srkf_predict_covariance(U, A, U_Q)

    return MvNormal(μ̂, PDMat(Cholesky(UpperTriangular(Û))))
end

function _srkf_predict_covariance(U, A, U_Q)
    M = vcat(U_Q, U * A')
    _, R = qr(M)
    return _correct_cholesky_sign(R)
end

"""
    srkf_update(state, obs_params, observation, jitter)

Perform the square-root Kalman filter update step.

Given the predicted state, observation parameters `(H, c, R)`, and observation `y`,
compute the filtered state and log-likelihood using QR factorization.
"""
function srkf_update(state::MvNormal, obs_params, observation, jitter)
    μ, Σ = params(state)
    H, c, R = obs_params

    U = cholesky(Σ).U
    U_R = cholesky(R).U

    μ̂, Û, ll = _srkf_update_covariance(μ, U, observation, H, c, U_R)

    if !isnothing(jitter)
        Û = Û + jitter * I
    end

    return MvNormal(μ̂, PDMat(Cholesky(UpperTriangular(Û)))), ll
end

function _srkf_update_covariance(μ, U, y, H, c, U_R)
    Dy = size(H, 1)
    Dx = size(H, 2)

    M = _srkf_form_update_matrix(U, H, U_R)
    _, R = qr(M)
    R = _correct_cholesky_sign(R)

    U_S, PHt, Û = _srkf_extract_update_components(R, Dx, Dy)

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

function _srkf_extract_update_components(R, Dx, Dy)
    U_S = R[1:Dy, 1:Dy]
    PHt = R[1:Dy, (Dy + 1):(Dy + Dx)]'
    Û = R[(Dy + 1):(Dy + Dx), (Dy + 1):(Dy + Dx)]
    return U_S, PHt, Û
end

function _srkf_loglikelihood(w, U_S, Dy)
    return -0.5 * (dot(w, w) + 2 * sum(log.(diag(U_S))) + Dy * log(2π))
end

# StaticArrays specializations for type stability

function _srkf_predict_covariance(
    U::UpperTriangular{T,<:SMatrix{Dx,Dx,T}},
    A::SMatrix{Dx,Dx,T},
    U_Q::UpperTriangular{T,<:SMatrix{Dx,Dx,T}},
) where {Dx,T}
    M = vcat(parent(U_Q), parent(U) * A')
    _, R = qr(M)
    return _correct_cholesky_sign(R)
end

function _srkf_form_update_matrix(
    U::UpperTriangular{T,<:SMatrix{Dx,Dx,T}},
    H::SMatrix{Dy,Dx,T},
    U_R::UpperTriangular{T,<:SMatrix{Dy,Dy,T}},
) where {Dx,Dy,T}
    top = hcat(parent(U_R), @SMatrix zeros(T, Dy, Dx))
    bottom = hcat(parent(U) * H', parent(U))
    return vcat(top, bottom)
end

function _srkf_extract_update_components(R::SMatrix{N,N,T}, Dx::Int, Dy::Int) where {N,T}
    U_S = R[SOneTo(Dy), SOneTo(Dy)]
    PHt = R[SOneTo(Dy), SUnitRange(Dy + 1, Dy + Dx)]'
    Û = R[SUnitRange(Dy + 1, Dy + Dx), SUnitRange(Dy + 1, Dy + Dx)]
    return U_S, PHt, Û
end
