"""
Mooncake.jl integration for the Kalman filter log-likelihood gradient.

This file implements a native Mooncake rrule!! for `kf_loglikelihood` using the
analytical gradient formulas from `kalman_gradient.jl`.

Note: We cannot use `@from_rrule` to wrap our ChainRulesCore.rrule because of a
tangent type mismatch. ChainRules returns plain `Matrix` gradients for `PDMat`
arguments, but Mooncake expects structured tangents (`Tangent{@NamedTuple{mat, chol}}`).
The native rrule handles this by manually placing gradients into the correct fields.

Limitation: This rrule currently only supports mutable array types (Vector, Matrix).
StaticArrays (SVector, SMatrix) are not yet supported due to Mooncake's NoFData
handling for immutable types.
"""

using Mooncake: Mooncake, @is_primitive, CoDual, primal, tangent, NoFData, NoRData, FData

# Helpers to increment tangents. Returns the updated tangent (may be mutated or new object).
# Mutable arrays: in-place addition
function _inc_tangent(t::AbstractArray, val)
    t .+= val
    return t
end

# FData for PDMat: the mat field is a mutable Matrix, mutate it directly
function _inc_tangent(t::FData, val)
    t.data.mat .+= val
    return t
end

# Tangent for PDMat: has mat and chol fields
function _inc_tangent(t::Mooncake.Tangent{<:NamedTuple{(:mat, :chol)}}, val)
    new_mat = _inc_tangent(t.fields.mat, val)
    return Mooncake.Tangent((mat=new_mat, chol=t.fields.chol))
end

# Helper for vector of tangents - handles reassignment for immutable tangents
function _inc_tangent_at!(tv::AbstractVector, idx::Integer, val)
    tv[idx] = _inc_tangent(tv[idx], val)
    return nothing
end

"""
    Mooncake.rrule!!(::CoDual{typeof(kf_loglikelihood)}, ...)

Native Mooncake reverse-mode AD rule for the Kalman filter log-likelihood.
The forward pass runs the KF with gradient caching; the pullback computes
analytical gradients using the backward recursion from `kalman_gradient.jl`.
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
) where {T<:Real}
    # Extract primals
    μ0_p, Σ0_p = primal(μ0), primal(Σ0)
    As_p, bs_p, Qs_p = primal(As), primal(bs), primal(Qs)
    Hs_p, cs_p, Rs_p = primal(Hs), primal(cs), primal(Rs)
    ys_p = primal(ys)

    # Extract tangent storage
    dμ0, dΣ0 = tangent(μ0), tangent(Σ0)
    dAs, dbs, dQs = tangent(As), tangent(bs), tangent(Qs)
    dHs, dcs, dRs = tangent(Hs), tangent(cs), tangent(Rs)

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
        state, Hs_p[1], cs_p[1], Rs_p[1], ys_p[1], nothing
    )
    ll += ll_inc
    caches = Vector{typeof(first_cache)}(undef, n)
    caches[1] = first_cache

    for t in 2:n
        μ_prevs[t], Σ_prevs[t] = params(state)
        state = kalman_predict(state, (As_p[t], bs_p[t], Qs_p[t]))
        state, ll_inc, caches[t] = _kalman_update_cached(
            state, Hs_p[t], cs_p[t], Rs_p[t], ys_p[t], nothing
        )
        ll += ll_inc
    end

    # Reverse pass closure
    function kf_loglikelihood_mooncake_pb(Δll)
        ∂μ, ∂Σ = zero(μ0_p), zero(As_p[1])

        for t in n:-1:1
            cache = caches[t]
            s = -Δll  # Convert from LL gradient to NLL gradient convention

            # Observation parameter gradients
            _inc_tangent_at!(dcs, t, s * gradient_c(∂μ, cache))
            _inc_tangent_at!(dHs, t, s * gradient_H(∂μ, ∂Σ, cache, cache.Σ_pred, Hs_p[t]))
            _inc_tangent_at!(dRs, t, s * gradient_R(∂μ, ∂Σ, cache))

            # Propagate through update step
            ∂μ_pred, ∂Σ_pred = backward_gradient_update(∂μ, ∂Σ, cache, Hs_p[t], Rs_p[t])

            # Dynamics parameter gradients
            _inc_tangent_at!(dbs, t, s * gradient_b(∂μ_pred))
            _inc_tangent_at!(
                dAs, t, s * gradient_A(∂μ_pred, ∂Σ_pred, μ_prevs[t], Σ_prevs[t], As_p[t])
            )
            _inc_tangent_at!(dQs, t, s * gradient_Q(∂Σ_pred))

            # Propagate through predict step
            ∂μ, ∂Σ = backward_gradient_predict(∂μ_pred, ∂Σ_pred, As_p[t])
        end

        # Initial state gradients
        _inc_tangent(dμ0, -Δll * ∂μ)
        _inc_tangent(dΣ0, -Δll * ∂Σ)

        # Return NoRData for all arguments (gradients accumulated in fdata)
        return ntuple(_ -> NoRData(), 10)
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
}
