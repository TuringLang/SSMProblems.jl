"""
Mooncake.jl extension for GeneralisedFilters.

This extension provides a native Mooncake rrule!! for `kf_loglikelihood` using the
analytical gradient formulas from `kalman_gradient.jl`.

Mutable tangent buffers accumulate in place; immutable tangent components return
through the pullback.
"""
module MooncakeExt

using GeneralisedFilters:
    GeneralisedFilters,
    kf_loglikelihood,
    kalman_predict,
    _kalman_state,
    _kalman_update_cached,
    gradient_c,
    gradient_H,
    gradient_R,
    gradient_b,
    gradient_A,
    gradient_Q,
    backward_gradient_update,
    backward_gradient_predict

using PDMats: PDMat, PDiagMat, ScalMat
using LinearAlgebra: diag, Diagonal, Symmetric, Hermitian, triu, tril, tr
using StaticArrays: StaticArray, SVector

using Mooncake: Mooncake, @is_primitive, CoDual, primal, tangent
using Mooncake: NoFData, NoRData, Tangent
using Mooncake: zero_tangent, zero_rdata, rdata, primal_to_tangent!!, increment!!

_tangent_array(t, p::Array) = t
function _tangent_array(t, p::SubArray)
    return view(_tangent_array(t.fields.parent, parent(p)), parentindices(p)...)
end
function _tangent_array(t, p)
    return throw(ArgumentError("Unsupported Mooncake array representation: $(typeof(p))"))
end

function _accumulate!!(t, grad, p::AbstractArray)
    _tangent_array(t, p) .+= grad
    return t
end
function _accumulate!!(t, grad, p::StaticArray{S,T}) where {S,T<:AbstractFloat}
    projected = primal_to_tangent!!(zero_tangent(p), typeof(p)(grad))
    return increment!!(t, projected)
end
_accumulate!!(t, grad, p::AbstractFloat) = t + typeof(p)(grad)

function _accumulate!!(t, grad, p::Union{Symmetric{<:Real},Hermitian{<:Real}})
    upper = p.uplo == 'U'
    # Each stored off-diagonal entry controls both triangles of the covariance.
    data = (upper ? triu(grad + grad', 1) : tril(grad + grad', -1)) + Diagonal(diag(grad))
    return Tangent((data=_accumulate!!(t.fields.data, data, parent(p)), uplo=t.fields.uplo))
end
function _accumulate!!(t, grad, p::Union{Diagonal,PDiagMat})
    return Tangent((diag=_accumulate!!(t.fields.diag, diag(grad), p.diag),))
end
function _accumulate!!(t, grad, p::PDMat)
    return Tangent((mat=_accumulate!!(t.fields.mat, grad, p.mat), chol=t.fields.chol))
end
function _accumulate!!(t, grad, p::ScalMat)
    return Tangent((
        dim=t.fields.dim, value=_accumulate!!(t.fields.value, tr(grad), p.value)
    ))
end

function _accumulate_at!!(t, i, grad, p::AbstractVector)
    data = _tangent_array(t, p)
    data[i] = _accumulate!!(data[i], grad, p[i])
    return t
end
function _accumulate_at!!(t, i, grad, p::SVector)
    data = Base.setindex(t.fields.data, _accumulate!!(t.fields.data[i], grad, p[i]), i)
    return Tangent((data=data,))
end

## MOONCAKE RRULE!! ############################################################################

"""
    Mooncake.rrule!!(::CoDual{typeof(kf_loglikelihood)}, ...)

Native Mooncake reverse-mode AD rule for the Kalman filter log-likelihood.
The forward pass runs the KF with gradient caching; the pullback computes
analytical gradients using the backward recursion from `kalman_gradient.jl`.

Supports dense arrays and their views, static arrays, and structured covariances.
`jitter` is a nondifferentiable constant.
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

    n = length(ys_p)

    # Forward pass with caching
    state = _kalman_state(μ0_p, Σ0_p)
    μ_prevs = Vector{typeof(state.μ)}(undef, n)
    # The state switches between dense and static storage when the prior and the dynamics
    # disagree on it, and `Cholesky` has no conversion between the two before Julia 1.12.
    # Only `∂Σ_pred * A * Σ_prev` reads this, so store the bare matrix.
    Σ_prevs = Vector{typeof(state.Σ.mat)}(undef, n)
    ll = zero(eltype(μ0_p))

    if n == 0
        zeros = map(
            zero_rdata, (μ0_p, Σ0_p, As_p, bs_p, Qs_p, Hs_p, cs_p, Rs_p, ys_p, jitter_p)
        )
        return CoDual(ll, NoFData()), _ -> (NoRData(), zeros...)
    end

    # First step to get concrete cache type
    μ_prevs[1], Σ_prevs[1] = state.μ, state.Σ.mat
    state = kalman_predict(state, (As_p[1], bs_p[1], Qs_p[1]))
    state, ll_inc, first_cache = _kalman_update_cached(
        state, Hs_p[1], cs_p[1], Rs_p[1], ys_p[1], jitter_p
    )
    ll += ll_inc
    caches = Vector{typeof(first_cache)}(undef, n)
    caches[1] = first_cache

    for t in 2:n
        μ_prevs[t], Σ_prevs[t] = state.μ, state.Σ.mat
        state = kalman_predict(state, (As_p[t], bs_p[t], Qs_p[t]))
        state, ll_inc, caches[t] = _kalman_update_cached(
            state, Hs_p[t], cs_p[t], Rs_p[t], ys_p[t], jitter_p
        )
        ll += ll_inc
    end

    # Reverse pass closure
    function kf_loglikelihood_mooncake_pb(Δll)
        t_μ0, t_Σ0, t_As, t_bs, t_Qs, t_Hs, t_cs, t_Rs, t_ys = map(
            x -> tangent(tangent(x), zero_rdata(primal(x))),
            (μ0, Σ0, As, bs, Qs, Hs, cs, Rs, ys),
        )
        ∂μ, ∂Σ = zero(μ0_p), zero(As_p[1])

        for t in n:-1:1
            cache = caches[t]
            s = -Δll  # Convert from LL gradient to NLL gradient convention

            # Observation parameter gradients (as full matrices)
            grad_c = s * gradient_c(∂μ, cache)
            grad_H = s * gradient_H(∂μ, ∂Σ, cache, cache.Σ_pred, Hs_p[t])
            grad_R = s * gradient_R(∂μ, ∂Σ, cache)

            t_cs = _accumulate_at!!(t_cs, t, grad_c, cs_p)
            t_ys = _accumulate_at!!(t_ys, t, -grad_c, ys_p)
            t_Hs = _accumulate_at!!(t_Hs, t, grad_H, Hs_p)
            t_Rs = _accumulate_at!!(t_Rs, t, grad_R, Rs_p)

            # Propagate through update step
            ∂μ_pred, ∂Σ_pred = backward_gradient_update(∂μ, ∂Σ, cache, Hs_p[t], Rs_p[t])

            # Dynamics parameter gradients
            grad_b = s * gradient_b(∂μ_pred)
            grad_A = s * gradient_A(∂μ_pred, ∂Σ_pred, μ_prevs[t], Σ_prevs[t], As_p[t])
            grad_Q = s * gradient_Q(∂Σ_pred)

            t_bs = _accumulate_at!!(t_bs, t, grad_b, bs_p)
            t_As = _accumulate_at!!(t_As, t, grad_A, As_p)
            t_Qs = _accumulate_at!!(t_Qs, t, grad_Q, Qs_p)

            # Propagate through predict step
            ∂μ, ∂Σ = backward_gradient_predict(∂μ_pred, ∂Σ_pred, As_p[t])
        end

        # Initial state gradients (scaled by -Δll for LL convention)
        grad_μ0 = -Δll * ∂μ
        grad_Σ0 = -Δll * ∂Σ

        t_μ0 = _accumulate!!(t_μ0, grad_μ0, μ0_p)
        t_Σ0 = _accumulate!!(t_Σ0, grad_Σ0, Σ0_p)
        return (
            NoRData(),
            map(rdata, (t_μ0, t_Σ0, t_As, t_bs, t_Qs, t_Hs, t_cs, t_Rs, t_ys))...,
            zero_rdata(jitter_p),
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

end # module MooncakeExt
