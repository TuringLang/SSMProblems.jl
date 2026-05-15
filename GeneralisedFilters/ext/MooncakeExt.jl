"""
Mooncake.jl extension for GeneralisedFilters.

Provides a native Mooncake `rrule!!` for `_kalman_step` using the analytical gradient
formulas in `kalman_gradient.jl`. The rrule's pullback re-runs the forward arithmetic
inline so that all intermediate values are captured as closure locals — no separate
cache struct.

Parameter wrappers `Fixed` and `TimeVarying` carry no differentiable state; their
`tangent_type` is declared as `NoTangent` so that Mooncake skips gradient tracking
through them. Active parameters (`FixedParametric`, `TimeVaryingParametric`) reach the
primitive as plain values, and the rrule's `_maybe_grad_X` helpers dispatch on the
primal type to skip the matrix arithmetic for inactive arguments at compile time.

Tangent plumbing for PDMat-family covariances mirrors the previous `kf_loglikelihood`
rrule. The rrule is restricted to immutable static-array primals; mutable
`Vector`/`Matrix` variants are not yet supported.
"""
module MooncakeExt

using GeneralisedFilters:
    GeneralisedFilters,
    _kalman_step,
    _val,
    Fixed,
    TimeVarying,
    _compute_innovation,
    _compute_innovation_cov,
    _compute_kalman_gain,
    _compute_joseph_update,
    _apply_jitter_and_wrap,
    backward_gradient_update,
    backward_gradient_predict,
    gradient_A,
    gradient_b,
    gradient_Q,
    gradient_H,
    gradient_c,
    gradient_R

using Distributions: MvNormal, logpdf
using PDMats: AbstractPDMat, PDMat, PDiagMat, X_A_Xt
using LinearAlgebra: Symmetric, diag, Diagonal
using StaticArrays: SArray

using Mooncake: Mooncake, @is_primitive, CoDual, primal
using Mooncake: NoFData, NoRData, Tangent, RData, NoTangent
using Mooncake: zero_tangent, rdata, primal_to_tangent!!

## TANGENT DECLARATIONS ########################################################################

Mooncake.tangent_type(::Type{<:Fixed}) = NoTangent
Mooncake.tangent_type(::Type{<:TimeVarying}) = NoTangent

## PDMAT TANGENT PLUMBING (input direction) ####################################################
#
# `_to_rdata` builds the Mooncake rdata for an input cotangent, given the input's
# primal and a "primal-shaped" gradient (a plain matrix or vector returned by the
# analytical gradient functions). Adapted from the previous `kf_loglikelihood` rrule.

_project_to_param(grad, _) = grad
_project_to_param(grad::AbstractMatrix, ::PDMat) = grad
_project_to_param(grad::AbstractMatrix, ::PDiagMat) = diag(grad)

_inner_primal(p::PDMat) = p.mat
_inner_primal(p::PDiagMat) = p.diag

function _wrap_pdmat_tangent(t_new, inner_tangent, ::PDMat)
    return Tangent((mat=inner_tangent, chol=t_new.fields.chol))
end
_wrap_pdmat_tangent(t_new, inner_tangent, ::PDiagMat) = Tangent((diag=inner_tangent,))

_to_rdata(::Any, ::NoTangent) = NoRData()

function _to_rdata(primal, grad)
    return rdata(primal_to_tangent!!(zero_tangent(primal), grad))
end

function _to_rdata(primal::AbstractPDMat, grad::AbstractMatrix)
    grad_param = _project_to_param(grad, primal)
    inner_primal = _inner_primal(primal)
    t_new = zero_tangent(primal)
    inner_t = primal_to_tangent!!(zero_tangent(inner_primal), grad_param)
    full_t = _wrap_pdmat_tangent(t_new, inner_t, primal)
    return rdata(full_t)
end

## RDATA -> PRIMAL-SHAPED GRADIENT EXTRACTION (output direction) ###############################
#
# `_from_rdata(rdata, primal)` converts the Mooncake rdata of an output cotangent back
# into a primal-shaped value (a plain matrix or vector). Implementation drills through
# the rdata's NamedTuple structure based on the primal's field layout. SArrays have a
# single `:data` field holding an `NTuple`; PDMats have `(:mat, :chol)` and delegate
# the inner `.mat` extraction recursively.

_from_rdata(t, _) = t
_from_rdata(t::RData, primal::SArray) = typeof(primal)(t.data.data)
_from_rdata(t::RData, primal::PDMat) = _from_rdata(t.data.mat, primal.mat)
_from_rdata(t::RData, primal::PDiagMat) = Diagonal(_from_rdata(t.data.diag, primal.diag))

## MAYBE-GRAD DISPATCH #########################################################################

_maybe_grad_A(::Fixed, _, _, _, _, _) = NoTangent()
function _maybe_grad_A(_, ∂μ_pred, ∂Σ_pred, μ_prev, Σ_prev, A)
    return gradient_A(∂μ_pred, ∂Σ_pred, μ_prev, Σ_prev, A)
end

_maybe_grad_b(::Fixed, _) = NoTangent()
_maybe_grad_b(_, ∂μ_pred) = gradient_b(∂μ_pred)

_maybe_grad_Q(::Fixed, _) = NoTangent()
_maybe_grad_Q(_, ∂Σ_pred) = gradient_Q(∂Σ_pred)

_maybe_grad_H(::Fixed, _, _, _, _, _, _, _, _, _, _, _) = NoTangent()
function _maybe_grad_H(_, ∂μ_filt, ∂Σ_filt, Δll, μ_pred, μ_filt, z, S, K, I_KH, Σ_pred, H)
    return gradient_H(∂μ_filt, ∂Σ_filt, Δll, μ_pred, μ_filt, z, S, K, I_KH, Σ_pred, H)
end

_maybe_grad_c(::Fixed, _, _, _, _, _) = NoTangent()
_maybe_grad_c(_, ∂μ_filt, Δll, z, S, K) = gradient_c(∂μ_filt, Δll, z, S, K)

_maybe_grad_R(::Fixed, _, _, _, _, _, _) = NoTangent()
function _maybe_grad_R(_, ∂μ_filt, ∂Σ_filt, Δll, z, S, K)
    return gradient_R(∂μ_filt, ∂Σ_filt, Δll, z, S, K)
end

## RRULE!! #####################################################################################

@is_primitive Mooncake.DefaultCtx Tuple{
    typeof(_kalman_step),
    AbstractVector{<:Real},
    AbstractPDMat{<:Real},
    Any,
    Any,
    Any,
    Any,
    Any,
    Any,
    AbstractVector{<:Real},
    Union{Nothing,Real},
}

function Mooncake.rrule!!(
    ::CoDual{typeof(_kalman_step)},
    μ_cd::CoDual{<:AbstractVector{T}},
    Σ_cd::CoDual{<:AbstractPDMat{T}},
    A_cd::CoDual,
    b_cd::CoDual,
    Q_cd::CoDual,
    H_cd::CoDual,
    c_cd::CoDual,
    R_cd::CoDual,
    y_cd::CoDual{<:AbstractVector{T}},
    jitter_cd::CoDual,
) where {T<:Real}
    μ_prev = primal(μ_cd)
    Σ_prev = primal(Σ_cd)
    A_p, b_p, Q_p = primal(A_cd), primal(b_cd), primal(Q_cd)
    H_p, c_p, R_p = primal(H_cd), primal(c_cd), primal(R_cd)
    y = primal(y_cd)
    jitter = primal(jitter_cd)

    A_v, b_v, Q_v = _val(A_p), _val(b_p), _val(Q_p)
    H_v, c_v, R_v = _val(H_p), _val(c_p), _val(R_p)

    # Forward pass — locals captured by the pullback closure
    μ_pred = A_v * μ_prev + b_v
    Σ_pred = PDMat(Symmetric(X_A_Xt(Σ_prev, A_v) + Q_v))
    z = _compute_innovation(μ_pred, H_v, c_v, y)
    S = _compute_innovation_cov(Σ_pred, H_v, R_v)
    K = _compute_kalman_gain(Σ_pred, H_v, S)
    I_KH, Σ_filt_raw = _compute_joseph_update(Σ_pred, K, H_v, R_v)
    μ_filt = μ_pred + K * z
    Σ_filt = _apply_jitter_and_wrap(Σ_filt_raw, jitter)
    ll = logpdf(MvNormal(z, S), zero(z))

    function _kalman_step_pb(∂out)
        ∂μ_filt = _from_rdata(∂out[1], μ_filt)
        ∂Σ_filt = _from_rdata(∂out[2], Σ_filt)
        Δll = ∂out[3]

        ∂μ_pred, ∂Σ_pred = backward_gradient_update(
            ∂μ_filt, ∂Σ_filt, Δll, z, S, I_KH, H_v, R_v
        )
        ∂μ_prev, ∂Σ_prev = backward_gradient_predict(∂μ_pred, ∂Σ_pred, A_v)

        ∂A = _maybe_grad_A(A_p, ∂μ_pred, ∂Σ_pred, μ_prev, Σ_prev, A_v)
        ∂b = _maybe_grad_b(b_p, ∂μ_pred)
        ∂Q = _maybe_grad_Q(Q_p, ∂Σ_pred)
        ∂H = _maybe_grad_H(
            H_p, ∂μ_filt, ∂Σ_filt, Δll, μ_pred, μ_filt, z, S, K, I_KH, Σ_pred, H_v
        )
        ∂c = _maybe_grad_c(c_p, ∂μ_filt, Δll, z, S, K)
        ∂R = _maybe_grad_R(R_p, ∂μ_filt, ∂Σ_filt, Δll, z, S, K)

        return (
            NoRData(),
            _to_rdata(μ_prev, ∂μ_prev),
            _to_rdata(Σ_prev, ∂Σ_prev),
            _to_rdata(A_p, ∂A),
            _to_rdata(b_p, ∂b),
            _to_rdata(Q_p, ∂Q),
            _to_rdata(H_p, ∂H),
            _to_rdata(c_p, ∂c),
            _to_rdata(R_p, ∂R),
            NoRData(),
            NoRData(),
        )
    end

    return CoDual((μ_filt, Σ_filt, ll), NoFData()), _kalman_step_pb
end

end # module MooncakeExt
