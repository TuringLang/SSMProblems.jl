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
using LinearAlgebra: Symmetric, Diagonal, eigen
using StaticArrays: SVector, SMatrix, StaticVector
import Mooncake as MC

## TANGENT HELPERS #########################################################################

# Rebuild a static array from the `data` field of its Mooncake tangent.
_mc_static_array(t, x) = typeof(x)(MC.get_tangent_field(t, :data))
# Build the Mooncake tangent of a static array from its entries.
_mc_static_tangent(x) = MC.build_tangent(typeof(x), Tuple(x))

# Seed the reverse pass: recover the cotangents of the filtered mean, covariance, and
# log-likelihood from the output codual's forward and reverse data.
function _mc_seed(dy_rdata, y_fdata, c)
    dy_state, dy_ll = MC.tangent(y_fdata, dy_rdata)
    dμ = _mc_static_array(MC.get_tangent_field(dy_state, :μ), c.μ0)
    dΣ = _mc_static_array(MC.get_tangent_field(dy_state, :Σ), c.Σ0)
    return dμ, dΣ, dy_ll
end

function _mc_state_tangent(state, g)
    return MC.build_tangent(
        typeof(state), _mc_static_tangent(g.μ0̄), _mc_static_tangent(g.Σ0̄)
    )
end

function _mc_dyn_tangent(dyn::LinearGaussianDynamics, g)
    return MC.build_tangent(
        typeof(dyn),
        _mc_static_tangent(g.Ā),
        _mc_static_tangent(g.b̄),
        _mc_static_tangent(g.Q̄),
    )
end
function _mc_obs_tangent(obs::LinearGaussianObservation, g)
    return MC.build_tangent(
        typeof(obs),
        _mc_static_tangent(g.H̄),
        _mc_static_tangent(g.c̄),
        _mc_static_tangent(g.R̄),
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
    GaussianState{<:SVector,<:SMatrix},
    MaybeWithFlags{<:LinearGaussianDynamics{<:SMatrix,<:SVector,<:SMatrix}},
    MaybeWithFlags{<:LinearGaussianObservation{<:SMatrix,<:SVector,<:SMatrix}},
    StaticVector,
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
        dμ, dΣ, dll = _mc_seed(dy_rdata, MC.tangent(out_cd), c)
        g = _kalman_adjoints(c, dμ, dΣ, dll, dyn, obs)
        return (
            MC.NoRData(),
            MC.rdata(_mc_state_tangent(state, g)),
            MC.rdata(_mc_dyn_tangent(dyn, g)),
            MC.rdata(_mc_obs_tangent(obs, g)),
            MC.zero_rdata(y),
        )
    end
    return out_cd, kalman_step_pullback!!
end

## EIGENVALUE-CLIPPING REPAIR PRIMITIVE ####################################################

# Divided differences of `f(λ) = max(λ, ε)` for the spectral-function pullback. Degenerate
# eigenvalues fall back to the derivative `f'(λ) = (λ > ε)` since `f` is piecewise linear.
function _clip_divided_differences(λ, ε)
    T = eltype(λ)
    tol = sqrt(eps(real(T)))
    f(x) = max(x, ε)
    fp(x) = x > ε ? one(T) : zero(T)
    kfun(li, lj) = abs(li - lj) > tol ? (f(li) - f(lj)) / (li - lj) : fp((li + lj) / 2)
    return kfun.(λ, λ')
end

MC.@is_primitive MC.DefaultCtx MC.ReverseMode Tuple{
    typeof(repair_covariance),EigenClip,SMatrix
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
        Σ̄⁺ = symmetrise(_mc_static_array(MC.tangent(MC.tangent(out_cd), dΣ⁺_rdata), Σ))
        Kmat = _clip_divided_differences(λ, clip.ε)
        Σ̄ = symmetrise(V * (Kmat .* (V' * Σ̄⁺ * V)) * V')
        return (MC.NoRData(), MC.zero_rdata(clip), MC.rdata(_mc_static_tangent(Σ̄)))
    end
    return out_cd, repair_pullback!!
end

end
