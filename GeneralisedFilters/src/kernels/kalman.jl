export CovarianceRepair, NoRepair, Jitter, EigenClip

## COVARIANCE REPAIR #######################################################################

"""
    CovarianceRepair

A positive-definiteness repair strategy applied to a filtered covariance after each update.
Repair sits outside the fused Kalman step so its pullback composes by the ordinary chain
rule.
"""
abstract type CovarianceRepair end

struct NoRepair <: CovarianceRepair end

"""Additive diagonal shift `Σ + ε I`."""
struct Jitter{T<:Real} <: CovarianceRepair
    ε::T
end

"""Eigenvalue clipping `Σ ↦ V max(Λ, ε) Vᵀ`."""
struct EigenClip{T<:Real} <: CovarianceRepair
    ε::T
end

repair_covariance(::NoRepair, Σ) = Σ
# `one(Σ)` keeps the identity as a static matrix rather than a UniformScaling.
repair_covariance(j::Jitter, Σ) = Σ + j.ε * one(Σ)
function repair_covariance(c::EigenClip, Σ)
    E = eigen(Symmetric(Σ))
    return symmetrise(E.vectors * Diagonal(max.(E.values, c.ε)) * E.vectors')
end

## KALMAN KERNELS ##########################################################################

function kalman_predict(state::GaussianState, d::LinearGaussianDynamics)
    μ̂ = d.A * state.μ + d.b
    Σ̂ = symmetrise(d.A * state.Σ * d.A' + d.Q)
    return GaussianState(μ̂, Σ̂)
end

"""
    kalman_update_cached(state, o, y)

Joseph-form Kalman update returning `(filtered_state, ll_increment, cache)`. The cache
exposes the intermediates consumed by the analytic reverse pass.
"""
function kalman_update_cached(state::GaussianState, o::LinearGaussianObservation, y)
    μ̂, Σ̂ = state.μ, state.Σ
    H, c, R = o.H, o.c, o.R

    v = y - (H * μ̂ + c)
    S = symmetrise(H * Σ̂ * H' + R)
    Sc = cholesky(Symmetric(S))
    Si = Sc \ one(S)
    K = Σ̂ * H' * Si
    IKH = I - K * H
    Σ = symmetrise(IKH * Σ̂ * IKH' + K * R * K')
    μ = μ̂ + K * v
    w = Si * v
    T = eltype(v)
    ll = -(length(c) * log(2 * T(π)) + logdet(Sc) + dot(v, w)) / 2

    cache = (; μ̂, Σ̂, H, v, S, Si, K, w)
    return GaussianState(μ, Σ), ll, cache
end

function kalman_update(
    state::GaussianState,
    o::LinearGaussianObservation,
    y;
    repair::CovarianceRepair=NoRepair(),
)
    filt, ll, _ = kalman_update_cached(state, o, y)
    return GaussianState(filt.μ, repair_covariance(repair, filt.Σ)), ll
end

"""
    kalman_step_cached(state, d, o, y)

Fused predict-and-update returning `(filtered_state, ll_increment, cache)`. This is the
differentiable primitive; the log-likelihood uses the pre-repair innovation covariance and
no repair is applied inside it.
"""
function kalman_step_cached(
    state::GaussianState, d::LinearGaussianDynamics, o::LinearGaussianObservation, y
)
    pred = kalman_predict(state, d)
    filt, ll, uc = kalman_update_cached(pred, o, y)
    cache = (;
        μ0=state.μ,
        Σ0=state.Σ,
        A=d.A,
        H=uc.H,
        μ̂=uc.μ̂,
        Σ̂=uc.Σ̂,
        v=uc.v,
        S=uc.S,
        Si=uc.Si,
        K=uc.K,
        w=uc.w,
    )
    return filt, ll, cache
end

# The fused step is the differentiable primitive (the Mooncake reverse rule is registered on
# it). Its arguments are left untyped so activity-flagged components pass straight through
# `_component`; the typed `kalman_step_cached` remains the kernel that rejects wrong types.
function kalman_step(state, dyn, obs, y)
    filt, ll, _ = kalman_step_cached(state, _component(dyn), _component(obs), y)
    return filt, ll
end

## RTS SMOOTHER KERNEL #####################################################################

"""
    rts_backward_step(filtered, d, smoothed_next, predicted=nothing)

Single Rauch-Tung-Striebel backward step. `predicted` is `p(x_{t+1} | y_{1:t})`; if omitted
it is recomputed from `filtered`.
"""
function rts_backward_step(
    filtered::GaussianState,
    d::LinearGaussianDynamics,
    smoothed_next::GaussianState,
    predicted::Union{Nothing,GaussianState}=nothing,
)
    pred = isnothing(predicted) ? kalman_predict(filtered, d) : predicted
    G = filtered.Σ * d.A' / cholesky(Symmetric(pred.Σ))
    μ = filtered.μ + G * (smoothed_next.μ - pred.μ)
    Σ = symmetrise(filtered.Σ + G * (smoothed_next.Σ - pred.Σ) * G')
    return GaussianState(μ, Σ)
end

## BACKWARD INFORMATION KERNELS ############################################################

"""
    BackwardInformationPredictor(; initial_jitter=nothing, jitter=nothing)

Recursively computes the predictive likelihood `p(y_{t:T} | x_t)` of a linear-Gaussian model
in information form. The jitter fields add numerical-stability shifts to the precision matrix
at initialisation and during backward prediction.

Based on https://arxiv.org/pdf/1505.06357.
"""
struct BackwardInformationPredictor{T0,T} <: AbstractBackwardPredictor
    initial_jitter::T0
    jitter::T
end
function BackwardInformationPredictor(; initial_jitter=nothing, jitter=nothing)
    return BackwardInformationPredictor(initial_jitter, jitter)
end

function backward_initialise(
    algo::BackwardInformationPredictor, o::LinearGaussianObservation, y
)
    H, c, R = o.H, o.c, o.R
    Rc = cholesky(Symmetric(R))
    λ = H' * (Rc \ (y - c))
    Ω = symmetrise(H' * (Rc \ H))
    if !isnothing(algo.initial_jitter)
        Ω = Ω + algo.initial_jitter * one(Ω)
    end
    return InformationLikelihood(λ, Ω)
end

function backward_predict(
    algo::BackwardInformationPredictor,
    lik::InformationLikelihood,
    d::LinearGaussianDynamics,
)
    λ, Ω = natural_params(lik)
    A, b, Q = d.A, d.b, d.Q
    # Solve the equivalent system rather than subtracting nearly equal matrices.
    # The square-root predictor remains preferable for ill-conditioned multivariate models.
    m = λ - Ω * b
    W = I + Ω * Q
    Ω̂ = symmetrise(A' * (W \ Ω) * A)
    λ̂ = A' * (W \ m)

    if !isnothing(algo.jitter)
        Ω̂ = Ω̂ + algo.jitter * one(Ω̂)
    end
    return InformationLikelihood(λ̂, Ω̂)
end

function backward_update(
    ::BackwardInformationPredictor,
    lik::InformationLikelihood,
    o::LinearGaussianObservation,
    y,
)
    λ, Ω = natural_params(lik)
    H, c, R = o.H, o.c, o.R
    Rc = cholesky(Symmetric(R))
    λ̂ = λ + H' * (Rc \ (y - c))
    Ω̂ = symmetrise(Ω + H' * (Rc \ H))
    return InformationLikelihood(λ̂, Ω̂)
end

## TWO-FILTER SMOOTHING ####################################################################

function two_filter_smooth(filtered::GaussianState, backward_lik::InformationLikelihood)
    μ_filt, Σ_filt = filtered.μ, filtered.Σ
    λ_back, Ω_back = natural_params(backward_lik)

    Σ_filt_inv = inv(Σ_filt)
    λ_filt = Σ_filt_inv * μ_filt

    Ω_smooth = symmetrise(Σ_filt_inv + Ω_back)
    λ_smooth = λ_filt + λ_back

    Σ_smooth = inv(Ω_smooth)
    μ_smooth = Σ_smooth * λ_smooth
    return GaussianState(μ_smooth, symmetrise(Σ_smooth))
end

"""
    compute_marginal_predictive_likelihood(forward, backward)

Marginal predictive likelihood `p(y_{t:T} | y_{1:t-1})` from a one-step predicted Gaussian
`forward` and a backward predictive likelihood `backward`. Based on Lemma 1 of
https://arxiv.org/pdf/1505.06357.
"""
function compute_marginal_predictive_likelihood(
    forward::GaussianState, backward::InformationLikelihood
)
    μ, Σ = forward.μ, forward.Σ
    λ, Ω = natural_params(backward)
    Γ = cholesky(Symmetric(Σ)).L

    Λ = symmetrise(Γ' * Ω * Γ + I)
    Λc = cholesky(Symmetric(Λ))
    M = Γ' * (λ - Ω * μ)
    ζ = dot(μ, Ω * μ) - 2 * dot(λ, μ) - dot(M, Λc \ M)
    return -(logdet(Λc) + ζ) / 2
end

export SqrtBackwardInformationPredictor, SqrtInformationLikelihood

"""
    SqrtInformationLikelihood(B, r, logscale)

Gaussian likelihood `exp(logscale - sum(abs2, B*z-r)/2)`. `B` may be rank deficient;
this is a likelihood, not a normalized distribution on `z`. Square-root backward
recursions retain its normalization constant, including constants from QR compression.
"""
struct SqrtInformationLikelihood{TB,TR,TC} <: AbstractLikelihood
    B::TB
    r::TR
    logscale::TC
end
natural_params(l::SqrtInformationLikelihood) = (l.B' * l.r, symmetrise(l.B' * l.B))

"""
    SqrtBackwardInformationPredictor()

Exact Gaussian backward likelihoods evaluated using QR and whitened residual factors.
Unlike information-matrix subtraction, this preserves weak and strong information without
cancellation. Explicit `CovarianceFactor`s support singular process noise. No covariance
or precision repair is applied. Backward messages are used for trajectory updates and are
not differentiated by the HMC parameter update.
"""
struct SqrtBackwardInformationPredictor <: AbstractBackwardPredictor end

function _compress_residual(B, r, logscale)
    n = size(B, 2)
    # Padding ensures square output even when there are fewer observations than states.
    M = vcat(hcat(B, r), zeros(eltype(B), n + 1, n + 1))
    R = qr(M).R
    return SqrtInformationLikelihood(
        R[1:n, 1:n], R[1:n, n + 1], logscale - abs2(R[n + 1, n + 1]) / 2
    )
end
function _compress_residual(B::StaticMatrix{M,N}, r::StaticVector, logscale) where {M,N}
    T = promote_type(eltype(B), eltype(r))
    R = qr(vcat(hcat(B, r), zero(SMatrix{N + 1,N + 1,T}))).R
    return SqrtInformationLikelihood(
        R[SOneTo(N), SOneTo(N)], R[SOneTo(N), N + 1], logscale - abs2(R[N + 1, N + 1]) / 2
    )
end

# Upper root of I + C*C', obtained without forming a potentially ill-conditioned Gram matrix.
function _identity_plus_root(C)
    return _correct_cholesky_sign(
        qr(vcat(Matrix{eltype(C)}(I, size(C, 1), size(C, 1)), C')).R
    )
end
function _identity_plus_root(C::StaticMatrix{M,N,T}) where {M,N,T}
    return _correct_cholesky_sign(qr(vcat(one(SMatrix{M,M,T}), C')).R)
end

function backward_initialise(
    ::SqrtBackwardInformationPredictor, o::LinearGaussianObservation, y
)
    L = _covariance_root(o.R)
    # Observation noise must be square and nonsingular for a normalized density.
    size(L, 1) == size(L, 2) ||
        throw(ArgumentError("observation noise factor must be square and nonsingular"))
    B, r = L \ o.H, L \ (y - o.c)
    logscale = -(length(y) * log(2π) + 2 * logabsdet(L)[1]) / 2
    return _compress_residual(B, r, logscale)
end
function backward_predict(
    ::SqrtBackwardInformationPredictor,
    l::SqrtInformationLikelihood,
    d::LinearGaussianDynamics,
)
    C = l.B * _covariance_root(d.Q)
    U = UpperTriangular(_identity_plus_root(C))
    B = U' \ (l.B * d.A)
    r = U' \ (l.r - l.B * d.b)
    logscale = l.logscale - sum(log, diag(U))
    return _compress_residual(B, r, logscale)
end
function backward_update(
    bp::SqrtBackwardInformationPredictor,
    l::SqrtInformationLikelihood,
    o::LinearGaussianObservation,
    y,
)
    obs = backward_initialise(bp, o, y)
    return _compress_residual(vcat(l.B, obs.B), vcat(l.r, obs.r), l.logscale + obs.logscale)
end

_forward_root(g::GaussianState) = _covariance_root(g.Σ)
_forward_root(g::SqrtGaussianState) = g.U'
"""
    compute_marginal_predictive_likelihood(forward, backward::SqrtInformationLikelihood; include_constant=false)

Gaussian overlap, omitting the suffix-common log normalization by default for accurate
relative ancestor weights. Set `include_constant=true` for a normalized suffix likelihood.
"""
function compute_marginal_predictive_likelihood(
    forward::Union{GaussianState,SqrtGaussianState},
    l::SqrtInformationLikelihood;
    include_constant::Bool=false,
)
    C = l.B * _forward_root(forward)
    U = UpperTriangular(_identity_plus_root(C))
    v = U' \ (l.r - l.B * forward.μ)
    return (include_constant ? l.logscale : zero(l.logscale)) - sum(log, diag(U)) -
           sum(abs2, v) / 2
end
