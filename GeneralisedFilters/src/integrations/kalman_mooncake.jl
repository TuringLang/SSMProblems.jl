"""
Mooncake.jl integration for the Kalman filter log-likelihood gradient.

This file implements a native Mooncake rrule!! for `kf_loglikelihood` using the
analytical gradient formulas from `kalman_gradient.jl`.

The implementation properly handles both mutable (Vector, Matrix) and immutable
(SVector, SMatrix) array types by respecting Mooncake's fdata/rdata separation:
- Mutable inputs: gradients accumulated in fdata, pullback returns NoRData
- Immutable inputs: pullback returns RData containing the gradient
"""

using Mooncake: Mooncake, @is_primitive, CoDual, primal, tangent
using Mooncake: NoFData, NoRData, RData, Tangent, FData
using Mooncake: zero_tangent, rdata, primal_to_tangent!!, increment!!
using PDMats: PDiagMat

## GRADIENT PROJECTION LAYER ###################################################################
#
# These functions project matrix gradients to parameter gradients. They are AD-framework
# agnostic and could eventually move to PDMats.jl as ChainRulesCore.ProjectTo implementations.

"""
    _project_to_param(grad_matrix, primal)

Project a gradient w.r.t. the full matrix representation to a gradient w.r.t. the
parameter representation.

For most types, this is identity. For structured covariance types like PDiagMat,
this extracts the relevant components (e.g., diagonal).
"""
_project_to_param(grad, primal) = grad
_project_to_param(grad::AbstractMatrix, ::PDMat) = grad
_project_to_param(grad::AbstractMatrix, ::PDiagMat) = diag(grad)

# TODO: Add ScalMat support if needed:
# _project_to_param(grad::AbstractMatrix, ::ScalMat) = sum(diag(grad))

## MOONCAKE TANGENT FIELD ACCESS ###############################################################
#
# These helpers encapsulate the tangent field structure for each PDMat type.
# Dispatch is on primal type, keeping tangent structure knowledge in one place.

"""
    _get_param_value(primal)

Return the underlying differentiable parameter for a PDMat type.
"""
_get_param_value(p::PDMat) = p.mat
_get_param_value(p::PDiagMat) = p.diag

"""
    _param_tangent(tangent, primal)

Extract the tangent for the differentiable parameter field from a PDMat tangent.
"""
_param_tangent(t::Tangent, ::PDMat) = t.fields.mat
_param_tangent(t::Tangent, ::PDiagMat) = t.fields.diag
_param_tangent(t::FData, ::PDMat) = t.data.mat
_param_tangent(t::FData, ::PDiagMat) = t.data.diag

"""
    _with_param_tangent(old_tangent, new_param_tangent, primal)

Reconstruct a PDMat tangent with an updated parameter field.
"""
function _with_param_tangent(t::Tangent, new_param, ::PDMat)
    return Tangent((mat=new_param, chol=t.fields.chol))
end
function _with_param_tangent(t::Tangent, new_param, ::PDiagMat)
    return Tangent((diag=new_param,))
end

## MOONCAKE TANGENT UPDATE #####################################################################

"""
    _update_vector_tangent!(fdata, idx, grad_matrix, primal_value)

Accumulate the gradient into the tangent at index `idx` in the fdata vector.
The gradient is first projected to parameter space, then converted to Mooncake
tangent format and accumulated.
"""
function _update_vector_tangent!(
    fdata::AbstractVector, idx::Integer, grad_matrix, primal_value
)
    grad_param = _project_to_param(grad_matrix, primal_value)
    grad_tangent = primal_to_tangent!!(zero_tangent(primal_value), grad_param)
    fdata[idx] = increment!!(fdata[idx], grad_tangent)
    return nothing
end

# Specialization for AbstractPDMat: update only the parameter field
function _update_vector_tangent!(
    fdata::AbstractVector, idx::Integer, grad_matrix, primal_value::AbstractPDMat
)
    grad_param = _project_to_param(grad_matrix, primal_value)
    param_value = _get_param_value(primal_value)

    old_tangent = fdata[idx]
    old_param_tangent = _param_tangent(old_tangent, primal_value)

    grad_tangent = primal_to_tangent!!(zero_tangent(param_value), grad_param)
    new_param_tangent = increment!!(old_param_tangent, grad_tangent)

    fdata[idx] = _with_param_tangent(old_tangent, new_param_tangent, primal_value)
    return nothing
end

## RDATA COMPUTATION ###########################################################################

"""
    _compute_rdata(grad_matrix, primal_value)

Compute the RData for an immutable input given its gradient (in matrix form).
Projects to parameter space, creates a tangent, and extracts the rdata portion.
"""
function _compute_rdata(grad_matrix, primal_value)
    grad_param = _project_to_param(grad_matrix, primal_value)
    t = primal_to_tangent!!(zero_tangent(primal_value), grad_param)
    return rdata(t)
end

# For AbstractPDMat: build tangent with only the parameter field set
function _compute_rdata(grad_matrix, primal_value::AbstractPDMat)
    grad_param = _project_to_param(grad_matrix, primal_value)
    param_value = _get_param_value(primal_value)

    t_new = zero_tangent(primal_value)
    t_param = primal_to_tangent!!(zero_tangent(param_value), grad_param)
    new_tangent = _with_param_tangent(t_new, t_param, primal_value)
    return rdata(new_tangent)
end

## FDATA UPDATE ################################################################################

"""
    _update_fdata!(fdata, grad_matrix, primal_value)

Update mutable fdata in-place with the gradient. Used for standalone (non-vector) args.
"""
function _update_fdata!(fdata::FData, grad_matrix, primal_value::AbstractPDMat)
    grad_param = _project_to_param(grad_matrix, primal_value)
    param_fdata = _param_tangent(fdata, primal_value)
    primal_to_tangent!!(param_fdata, grad_param)
    return nothing
end

## MOONCAKE RRULE!! ############################################################################

"""
    Mooncake.rrule!!(::CoDual{typeof(kf_loglikelihood)}, ...)

Native Mooncake reverse-mode AD rule for the Kalman filter log-likelihood.
The forward pass runs the KF with gradient caching; the pullback computes
analytical gradients using the backward recursion from `kalman_gradient.jl`.

Supports both mutable (Vector, Matrix, PDMat{Matrix}) and immutable
(SVector, SMatrix, PDMat{SMatrix}) array types, as well as diagonal
covariance types (PDiagMat).
"""
function Mooncake.rrule!!(
    ::CoDual{typeof(kf_loglikelihood)},
    μ0::CoDual{<:AbstractVector{T}},
    Σ0::CoDual,
    As::CoDual{<:AbstractVector},
    bs::CoDual{<:AbstractVector},
    Qs::CoDual{<:AbstractVector},
    Hs::CoDual{<:AbstractVector},
    cs::CoDual{<:AbstractVector},
    Rs::CoDual{<:AbstractVector},
    ys::CoDual{<:AbstractVector},
    jitter::CoDual{<:Union{Nothing,Real}},
) where {T<:Real}
    # Extract primals
    μ0_p, Σ0_p = primal(μ0), primal(Σ0)
    As_p, bs_p, Qs_p = primal(As), primal(bs), primal(Qs)
    Hs_p, cs_p, Rs_p = primal(Hs), primal(cs), primal(Rs)
    ys_p = primal(ys)
    jitter_p = primal(jitter)

    # Extract tangent storage (fdata for mutable, NoFData for immutable)
    t_μ0, t_Σ0 = tangent(μ0), tangent(Σ0)
    t_As, t_bs, t_Qs = tangent(As), tangent(bs), tangent(Qs)
    t_Hs, t_cs, t_Rs = tangent(Hs), tangent(cs), tangent(Rs)

    n = length(ys_p)

    # Forward pass with caching
    state = MvNormal(μ0_p, Σ0_p)
    μ_prevs = Vector{typeof(μ0_p)}(undef, n)
    Σ_prevs = Vector{typeof(Σ0_p)}(undef, n)
    ll = zero(eltype(μ0_p))

    # First step to get concrete cache type
    μ_prevs[1], Σ_prevs[1] = params(state)
    state = kalman_predict(state, (As_p[1], bs_p[1], Qs_p[1]))
    state, ll_inc, first_cache = _kalman_update_cached(
        state, Hs_p[1], cs_p[1], Rs_p[1], ys_p[1], jitter_p
    )
    ll += ll_inc
    caches = Vector{typeof(first_cache)}(undef, n)
    caches[1] = first_cache

    for t in 2:n
        μ_prevs[t], Σ_prevs[t] = params(state)
        state = kalman_predict(state, (As_p[t], bs_p[t], Qs_p[t]))
        state, ll_inc, caches[t] = _kalman_update_cached(
            state, Hs_p[t], cs_p[t], Rs_p[t], ys_p[t], jitter_p
        )
        ll += ll_inc
    end

    # Reverse pass closure
    function kf_loglikelihood_mooncake_pb(Δll)
        ∂μ, ∂Σ = zero(μ0_p), zero(As_p[1])

        for t in n:-1:1
            cache = caches[t]
            s = -Δll  # Convert from LL gradient to NLL gradient convention

            # Observation parameter gradients (as full matrices)
            grad_c = s * gradient_c(∂μ, cache)
            grad_H = s * gradient_H(∂μ, ∂Σ, cache, cache.Σ_pred, Hs_p[t])
            grad_R = s * gradient_R(∂μ, ∂Σ, cache)

            _update_vector_tangent!(t_cs, t, grad_c, cs_p[t])
            _update_vector_tangent!(t_Hs, t, grad_H, Hs_p[t])
            _update_vector_tangent!(t_Rs, t, grad_R, Rs_p[t])

            # Propagate through update step
            ∂μ_pred, ∂Σ_pred = backward_gradient_update(∂μ, ∂Σ, cache, Hs_p[t], Rs_p[t])

            # Dynamics parameter gradients
            grad_b = s * gradient_b(∂μ_pred)
            grad_A = s * gradient_A(∂μ_pred, ∂Σ_pred, μ_prevs[t], Σ_prevs[t], As_p[t])
            grad_Q = s * gradient_Q(∂Σ_pred)

            _update_vector_tangent!(t_bs, t, grad_b, bs_p[t])
            _update_vector_tangent!(t_As, t, grad_A, As_p[t])
            _update_vector_tangent!(t_Qs, t, grad_Q, Qs_p[t])

            # Propagate through predict step
            ∂μ, ∂Σ = backward_gradient_predict(∂μ_pred, ∂Σ_pred, As_p[t])
        end

        # Initial state gradients (scaled by -Δll for LL convention)
        grad_μ0 = -Δll * ∂μ
        grad_Σ0 = -Δll * ∂Σ

        # Handle μ0: check if mutable or immutable
        rdata_μ0 = if t_μ0 isa NoFData
            _compute_rdata(grad_μ0, μ0_p)
        else
            primal_to_tangent!!(t_μ0, grad_μ0)
            NoRData()
        end

        # Handle Σ0: check if mutable or immutable
        rdata_Σ0 = if t_Σ0 isa NoFData
            _compute_rdata(grad_Σ0, Σ0_p)
        else
            _update_fdata!(t_Σ0, grad_Σ0, Σ0_p)
            NoRData()
        end

        # Vector arguments: always mutable containers, return NoRData
        # (the individual element tangents were updated in place above)
        return (
            NoRData(),   # function
            rdata_μ0,    # μ0
            rdata_Σ0,    # Σ0
            NoRData(),   # As
            NoRData(),   # bs
            NoRData(),   # Qs
            NoRData(),   # Hs
            NoRData(),   # cs
            NoRData(),   # Rs
            NoRData(),   # ys
            NoRData(),   # jitter (non-differentiable constant)
        )
    end

    return CoDual(ll, NoFData()), kf_loglikelihood_mooncake_pb
end

# Declare kf_loglikelihood as primitive for Mooncake
@is_primitive Mooncake.DefaultCtx Tuple{
    typeof(kf_loglikelihood),
    AbstractVector{<:Real},
    Any,
    AbstractVector,
    AbstractVector,
    AbstractVector,
    AbstractVector,
    AbstractVector,
    AbstractVector,
    AbstractVector,
    Union{Nothing,Real},
}
