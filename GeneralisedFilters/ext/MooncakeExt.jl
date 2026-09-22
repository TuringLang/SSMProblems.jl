module MooncakeExt

using GeneralisedFilters
using GeneralisedFilters:
    GaussianState,
    LinearGaussianDynamics,
    LinearGaussianObservation,
    EigenClip,
    WithFlags,
    MaybeWithFlags,
    _component,
    _kalman_adjoints,
    kalman_step,
    kalman_step_cached,
    repair_covariance,
    symmetrise
using LinearAlgebra: LinearAlgebra
using LinearAlgebra: Symmetric, Diagonal, eigen, triu, diag
using StaticArrays: SVector, SMatrix
import Mooncake as MC

# Julia's small dense matrix kernels inspect a structural wrapper flag through
# isuppercase. Its Unicode foreigncall has no numerical derivative; keep the
# matrix arithmetic itself on Mooncake's ordinary derived path. In particular,
# this preserves selected-triangle and shared-parent accumulation semantics.
MC.@zero_derivative MC.DefaultCtx Tuple{typeof(isuppercase),LinearAlgebra.WrapperChar}

## TANGENT HELPERS #########################################################################

# Rebuild a static array from the `data` field of its Mooncake tangent.
_mc_static_array(t, x) = typeof(x)(MC.get_tangent_field(t, :data))
# Build the Mooncake tangent of a static array from its entries.
_mc_static_tangent(x) = MC.build_tangent(typeof(x), Tuple(x))
_mc_static_tangent(x, primal) = _mc_static_tangent(typeof(primal)(x))

# Restrict the handwritten rule to immutable floating-point storage. Mutable static
# arrays and other scalar types use Mooncake's derived rules.
const FloatVector = SVector{N,T} where {N,T<:Union{Float32,Float64}}
const FloatMatrix = SMatrix{M,N,T,L} where {M,N,T<:Union{Float32,Float64},L}

# Seed the reverse pass: recover the cotangents of the filtered mean, covariance, and
# log-likelihood from the output codual's forward and reverse data.
function _mc_seed(dy_rdata, y_fdata, output)
    dy_state, dy_ll = MC.tangent(y_fdata, dy_rdata)
    dμ = _mc_static_array(MC.get_tangent_field(dy_state, :μ), output.μ)
    dΣ = _mc_static_array(MC.get_tangent_field(dy_state, :Σ), output.Σ)
    return dμ, dΣ, dy_ll
end

function _mc_state_tangent(state, g)
    return MC.build_tangent(
        typeof(state), _mc_static_tangent(g.μ0̄, state.μ), _mc_static_tangent(g.Σ0̄, state.Σ)
    )
end

function _mc_dyn_tangent(dyn::LinearGaussianDynamics, g)
    return MC.build_tangent(
        typeof(dyn),
        _mc_static_tangent(g.Ā, dyn.A),
        _mc_static_tangent(g.b̄, dyn.b),
        _mc_static_tangent(g.Q̄, dyn.Q),
    )
end
function _mc_obs_tangent(obs::LinearGaussianObservation, g)
    return MC.build_tangent(
        typeof(obs),
        _mc_static_tangent(g.H̄, obs.H),
        _mc_static_tangent(g.c̄, obs.c),
        _mc_static_tangent(g.R̄, obs.R),
    )
end

# A wrapped component's cotangent mirrors the `WithFlags` nesting.
function _mc_dyn_tangent(dyn::WithFlags, g)
    return MC.build_tangent(typeof(dyn), _mc_dyn_tangent(dyn.component, g))
end
function _mc_obs_tangent(obs::WithFlags, g)
    return MC.build_tangent(typeof(obs), _mc_obs_tangent(obs.component, g))
end

## FUSED KALMAN STEP PRIMITIVE #############################################################

# The primitive covers isbits static-array models only; heap-array models fall back to
# Mooncake's derived rules.
MC.@is_primitive MC.DefaultCtx MC.ReverseMode Tuple{
    typeof(kalman_step),
    GaussianState{<:FloatVector,<:FloatMatrix},
    MaybeWithFlags{<:LinearGaussianDynamics{<:FloatMatrix,<:FloatVector,<:FloatMatrix}},
    MaybeWithFlags{<:LinearGaussianObservation{<:FloatMatrix,<:FloatVector,<:FloatMatrix}},
    FloatVector,
}

function MC.rrule!!(
    ::MC.CoDual{typeof(kalman_step)},
    state_cd::MC.CoDual{<:GaussianState},
    dyn_cd::MC.CoDual,
    obs_cd::MC.CoDual,
    y_cd::MC.CoDual,
)
    state = MC.primal(state_cd)
    dyn = MC.primal(dyn_cd)
    obs = MC.primal(obs_cd)
    y = MC.primal(y_cd)
    new_state, ll, c = kalman_step_cached(state, _component(dyn), _component(obs), y)
    out_cd = MC.zero_fcodual((new_state, ll))

    function kalman_step_pullback!!(dy_rdata)
        dμ, dΣ, dll = _mc_seed(dy_rdata, MC.tangent(out_cd), new_state)
        g = _kalman_adjoints(c, dμ, dΣ, dll, dyn, obs)
        return (
            MC.NoRData(),
            MC.rdata(_mc_state_tangent(state, g)),
            MC.rdata(_mc_dyn_tangent(dyn, g)),
            MC.rdata(_mc_obs_tangent(obs, g)),
            MC.rdata(_mc_static_tangent(-dll * c.w + c.K' * dμ, y)),
        )
    end
    return out_cd, kalman_step_pullback!!
end

## EIGENVALUE-CLIPPING REPAIR PRIMITIVE ####################################################

# Divided differences of `f(λ) = max(λ, ε)` for the spectral-function pullback. Degenerate
# eigenvalues fall back to the derivative `f'(λ) = (λ > ε)` since `f` is piecewise linear.
function _clip_divided_differences(λ, ε)
    function kfun(li, lj)
        if li > ε && lj > ε
            return one(li)
        elseif li <= ε && lj <= ε
            return zero(li)
        else
            return (max(li, ε) - max(lj, ε)) / (li - lj)
        end
    end
    return kfun.(λ, λ')
end

MC.@is_primitive MC.DefaultCtx MC.ReverseMode Tuple{
    typeof(repair_covariance),EigenClip{<:Union{Float32,Float64}},FloatMatrix
}

function MC.rrule!!(
    ::MC.CoDual{typeof(repair_covariance)},
    clip_cd::MC.CoDual{<:EigenClip},
    Σ_cd::MC.CoDual{<:SMatrix},
)
    clip = MC.primal(clip_cd)
    Σ = MC.primal(Σ_cd)
    E = eigen(Symmetric(Σ))
    V, λ = E.vectors, E.values
    Σ⁺ = symmetrise(V * Diagonal(max.(λ, clip.ε)) * V')
    out_cd = MC.zero_fcodual(Σ⁺)

    function repair_pullback!!(dΣ⁺_rdata)
        Σ̄⁺ = symmetrise(_mc_static_array(MC.tangent(MC.tangent(out_cd), dΣ⁺_rdata), Σ⁺))
        Kmat = _clip_divided_differences(λ, clip.ε)
        Σ̄ = symmetrise(V * (Kmat .* (V' * Σ̄⁺ * V)) * V')
        Σ̄_storage = 2 * triu(Σ̄, 1) + Diagonal(diag(Σ̄))
        ε̄ = sum(diag(V' * Σ̄⁺ * V) .* (λ .<= clip.ε))
        clip_tangent = MC.build_tangent(typeof(clip), typeof(clip.ε)(ε̄))
        return (
            MC.NoRData(), MC.rdata(clip_tangent), MC.rdata(_mc_static_tangent(Σ̄_storage, Σ))
        )
    end
    return out_cd, repair_pullback!!
end

## DENSE SQUARE-ROOT QR PRIMITIVE ###########################################################

# Only R is needed by the filter. Avoid differentiating LAPACK's blocked QR foreigncall.
MC.@is_primitive MC.DefaultCtx MC.ReverseMode Tuple{
    typeof(GeneralisedFilters._qr_upper),Matrix{<:Union{Float32,Float64}}
}
function MC.rrule!!(
    ::MC.CoDual{typeof(GeneralisedFilters._qr_upper)}, M_cd::MC.CoDual{<:Matrix}
)
    M = MC.primal(M_cd)
    R = GeneralisedFilters._qr_upper(M)
    out = MC.zero_fcodual(R)
    function qr_upper_pullback!!(::MC.NoRData)
        G = MC.tangent(out)
        if any(!iszero, G)
            any(iszero, diag(R)) && throw(
                ArgumentError(
                    "reverse differentiation through a rank-deficient square-root QR is unsupported; use a fixed full-rank parameterization",
                ),
            )
            # R-only QR adjoint: Q*sym_upper(G*R')/R'.
            U = LinearAlgebra.UpperTriangular(R)
            D = G * R'
            S = Symmetric(D, :U)
            MC.tangent(M_cd) .+= (M / U) * S / U'
        end
        return MC.NoRData(), MC.NoRData()
    end
    return out, qr_upper_pullback!!
end

end
