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
    FixedParametric,
    TimeVarying,
    TimeVaryingParametric,
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
    gradient_R,
    _ssm_loglikelihood,
    _step_forward,
    _step_pullback,
    _step_initial,
    _initial_pullback,
    hoist_controls,
    hoist_static,
    resolve_controls,
    step_params,
    KalmanFilter,
    GaussianPrior,
    LinearGaussianLatentDynamics,
    LinearGaussianObservationProcess,
    LinearGaussianStateSpaceModel
using SSMProblems: prior, dyn, obs

using Distributions: MvNormal, logpdf, params
using PDMats: AbstractPDMat, PDMat, PDiagMat, X_A_Xt
using LinearAlgebra: Symmetric, diag, Diagonal
using StaticArrays: SArray

using Mooncake: Mooncake, @is_primitive, CoDual, primal
using Mooncake: NoFData, NoRData, Tangent, RData, NoTangent, NoCache
using Mooncake: zero_tangent, zero_fcodual, rdata, fdata, primal_to_tangent!!
using Mooncake: increment_internal!!, build_rrule

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

## TANGENT DECLARATIONS FOR PARAMETRIC WRAPPERS #################################################
# These never carry a tangent on their captures from Mooncake's perspective — captures'
# gradient contributions are recovered manually via `build_rrule` inside the rrule below.

Mooncake.tangent_type(::Type{<:FixedParametric}) = NoTangent
Mooncake.tangent_type(::Type{<:TimeVaryingParametric}) = NoTangent

## SSM_LOGLIKELIHOOD RRULE ######################################################################
#
# Strategy:
# - Forward pass mirrors `_ssm_loglikelihood`, additionally caching per-step intermediates
#   (state cache + resolved controls) so the backward sweep can reconstruct.
# - Backward sweep:
#   - Walks steps in reverse, calling `_step_pullback` to get per-step cotangents
#     for dyn / obs parameters. Each cotangent is routed by the parameter's trait:
#       * Fixed / TimeVarying  -> NoTangent (no work — `_step_pullback` elided arithmetic)
#       * FixedParametric      -> accumulate ∂value into a per-field buffer
#       * TimeVaryingParametric-> per-step Mooncake pullback through `p.f(θ, t, resolved)`
#   - After the loop, `_initial_pullback` produces ∂prior_params, routed similarly
#     (FixedParametric only — TVP for prior is a programming error).
#   - Finally, each FixedParametric param/control gets a single Mooncake pullback
#     through `p.f(θ, hoisted_controls)` (params) or `c.f(θ)` (controls) with the
#     accumulated buffer.
# - The outer θ_cd is passed through to ALL inner pullbacks so mutable fdata
#   contributions accumulate into a single buffer. Rdata contributions (for immutable
#   θ) are accumulated separately via `_add_rdata`.
#
# Note: this commit handles the case where parametric parameters depend on θ directly
# (and on time-varying controls via the trait dispatch). Parametric parameters that
# read parametric controls (the "shared θ-dependent computation" pattern) are NOT yet
# supported — those gradient paths require routing ∂hoisted_controls / ∂resolved back
# through the control closures, which is deferred to a follow-up commit.

# Rdata accumulation: handle NoRData no-op explicitly; otherwise delegate.
_add_rdata(::NoRData, ::NoRData) = NoRData()
_add_rdata(a::NoRData, b) = b
_add_rdata(a, ::NoRData) = a
_add_rdata(a, b) = increment_internal!!(NoCache(), a, b)

# Zero-cotangent for state at output. We differentiate ll, not the final state.
function _zero_state_cotangent(::KalmanFilter, state::MvNormal)
    μ, Σ = params(state)
    Σ_inner = Σ isa PDMat ? Σ.mat : Matrix(Σ)
    return (zero(μ), zero(Σ_inner))
end

# FixedParametric per-field accumulator buffer. `Ref{Any}` keeps init lazy so we don't
# have to know the cotangent type up front (relevant for PDMat-typed params).
_make_fp_buffer(::Any) = nothing
_make_fp_buffer(::FixedParametric) = Base.RefValue{Any}(nothing)

# Accumulate per-step cotangent into FP buffer (mutates).
_acc_fp!(::Nothing, _) = nothing
function _acc_fp!(buf::Base.RefValue{Any}, ∂val)
    buf[] = buf[] === nothing ? ∂val : buf[] + ∂val
    return nothing
end

# Accumulate the inner CoDual's fdata into the outer's. NoFData (immutable types)
# is a no-op; for mutable types this is an in-place increment.
_accumulate_fdata!(::NoFData, ::NoFData) = nothing
_accumulate_fdata!(outer, inner) = (increment_internal!!(NoCache(), outer, inner); nothing)

# Per-step Mooncake pullback through a TimeVaryingParametric closure `f(θ, t, resolved)`.
# Uses a FRESH inner θ CoDual per call (sharing the outer θ_cd's fdata across multiple
# rule invocations triggers a Mooncake bug for closures involving array slicing). After
# the pullback, the inner fdata is added into the outer θ_cd's fdata explicitly.
function _tvp_step_pullback!(f, ∂val, θ_cd, t, resolved)
    θ = primal(θ_cd)
    rule = build_rrule(Tuple{typeof(f),typeof(θ),typeof(t),typeof(resolved)})
    θ_inner = zero_fcodual(θ)
    out_cd, pb = rule(
        zero_fcodual(f), θ_inner, zero_fcodual(t), zero_fcodual(resolved)
    )
    out_primal = primal(out_cd)
    ∂val_tan = primal_to_tangent!!(zero_tangent(out_primal), ∂val)
    increment_internal!!(NoCache(), out_cd.dx, fdata(∂val_tan))
    _, ∂θ_rd, _, _ = pb(rdata(∂val_tan))
    _accumulate_fdata!(θ_cd.dx, θ_inner.dx)
    return ∂θ_rd
end

# End-of-loop Mooncake pullback through a FixedParametric closure `f(θ, hoisted_controls)`.
function _fp_finish_pullback!(f, ∂val, θ_cd, hoisted_controls)
    θ = primal(θ_cd)
    rule = build_rrule(Tuple{typeof(f),typeof(θ),typeof(hoisted_controls)})
    θ_inner = zero_fcodual(θ)
    out_cd, pb = rule(
        zero_fcodual(f), θ_inner, zero_fcodual(hoisted_controls)
    )
    out_primal = primal(out_cd)
    ∂val_tan = primal_to_tangent!!(zero_tangent(out_primal), ∂val)
    increment_internal!!(NoCache(), out_cd.dx, fdata(∂val_tan))
    _, ∂θ_rd, _ = pb(rdata(∂val_tan))
    _accumulate_fdata!(θ_cd.dx, θ_inner.dx)
    return ∂θ_rd
end

# Per-field step routing. Dispatches on the parameter wrapper's trait.
_route_param!(_, ∂θ_rd, ::Fixed, _, _, _, _) = ∂θ_rd
_route_param!(_, ∂θ_rd, ::TimeVarying, _, _, _, _) = ∂θ_rd
function _route_param!(buf, ∂θ_rd, ::FixedParametric, ∂val, _, _, _)
    _acc_fp!(buf, ∂val)
    return ∂θ_rd
end
function _route_param!(_, ∂θ_rd, p::TimeVaryingParametric, ∂val, θ_cd, t, resolved)
    return _add_rdata(∂θ_rd, _tvp_step_pullback!(p.f, ∂val, θ_cd, t, resolved))
end

# Component-level step routing for the LG dynamics / observation.
function _route_dyn_step!(bufs, ∂θ_rd, dyn::LinearGaussianLatentDynamics, ∂vals, θ_cd, t, resolved)
    ∂θ_rd = _route_param!(bufs.A, ∂θ_rd, dyn.A, ∂vals.A, θ_cd, t, resolved)
    ∂θ_rd = _route_param!(bufs.b, ∂θ_rd, dyn.b, ∂vals.b, θ_cd, t, resolved)
    ∂θ_rd = _route_param!(bufs.Q, ∂θ_rd, dyn.Q, ∂vals.Q, θ_cd, t, resolved)
    return ∂θ_rd
end

function _route_obs_step!(bufs, ∂θ_rd, obs::LinearGaussianObservationProcess, ∂vals, θ_cd, t, resolved)
    ∂θ_rd = _route_param!(bufs.H, ∂θ_rd, obs.H, ∂vals.H, θ_cd, t, resolved)
    ∂θ_rd = _route_param!(bufs.c, ∂θ_rd, obs.c, ∂vals.c, θ_cd, t, resolved)
    ∂θ_rd = _route_param!(bufs.R, ∂θ_rd, obs.R, ∂vals.R, θ_cd, t, resolved)
    return ∂θ_rd
end

# Prior routing (no time index; only FixedParametric makes sense).
_route_prior_field!(_, ∂θ_rd, ::Fixed, _, _, _) = ∂θ_rd
_route_prior_field!(_, ∂θ_rd, ::TimeVarying, _, _, _) = ∂θ_rd
function _route_prior_field!(buf, ∂θ_rd, ::FixedParametric, ∂val, _, _)
    _acc_fp!(buf, ∂val)
    return ∂θ_rd
end
function _route_prior_field!(_, _, ::TimeVaryingParametric, _, _, _)
    return error(
        "TimeVaryingParametric is not valid for prior parameters; use FixedParametric"
    )
end

function _route_prior!(bufs, ∂θ_rd, prior::GaussianPrior, ∂vals, θ_cd, hoisted_controls)
    ∂θ_rd = _route_prior_field!(bufs.μ0, ∂θ_rd, prior.μ0, ∂vals.μ0, θ_cd, hoisted_controls)
    ∂θ_rd = _route_prior_field!(bufs.Σ0, ∂θ_rd, prior.Σ0, ∂vals.Σ0, θ_cd, hoisted_controls)
    return ∂θ_rd
end

# End-of-loop FixedParametric pullback per field.
_finish_fp_field!(∂θ_rd, _, ::Nothing, _, _) = ∂θ_rd
function _finish_fp_field!(∂θ_rd, p::FixedParametric, buf::Base.RefValue{Any}, θ_cd, hoisted_controls)
    buf[] === nothing && return ∂θ_rd  # nothing accumulated (e.g. unused parameter)
    return _add_rdata(∂θ_rd, _fp_finish_pullback!(p.f, buf[], θ_cd, hoisted_controls))
end

function _finish_dyn_fp!(∂θ_rd, dyn::LinearGaussianLatentDynamics, bufs, θ_cd, hoisted_controls)
    ∂θ_rd = _finish_fp_field!(∂θ_rd, dyn.A, bufs.A, θ_cd, hoisted_controls)
    ∂θ_rd = _finish_fp_field!(∂θ_rd, dyn.b, bufs.b, θ_cd, hoisted_controls)
    ∂θ_rd = _finish_fp_field!(∂θ_rd, dyn.Q, bufs.Q, θ_cd, hoisted_controls)
    return ∂θ_rd
end

function _finish_obs_fp!(∂θ_rd, obs::LinearGaussianObservationProcess, bufs, θ_cd, hoisted_controls)
    ∂θ_rd = _finish_fp_field!(∂θ_rd, obs.H, bufs.H, θ_cd, hoisted_controls)
    ∂θ_rd = _finish_fp_field!(∂θ_rd, obs.c, bufs.c, θ_cd, hoisted_controls)
    ∂θ_rd = _finish_fp_field!(∂θ_rd, obs.R, bufs.R, θ_cd, hoisted_controls)
    return ∂θ_rd
end

function _finish_prior_fp!(∂θ_rd, prior::GaussianPrior, bufs, θ_cd, hoisted_controls)
    ∂θ_rd = _finish_fp_field!(∂θ_rd, prior.μ0, bufs.μ0, θ_cd, hoisted_controls)
    ∂θ_rd = _finish_fp_field!(∂θ_rd, prior.Σ0, bufs.Σ0, θ_cd, hoisted_controls)
    return ∂θ_rd
end

@is_primitive Mooncake.DefaultCtx Tuple{
    typeof(_ssm_loglikelihood),
    KalmanFilter,
    LinearGaussianStateSpaceModel,
    Any,
    AbstractVector,
    NamedTuple,
}

function Mooncake.rrule!!(
    ::CoDual{typeof(_ssm_loglikelihood)},
    filter_cd::CoDual{<:KalmanFilter},
    model_cd::CoDual{<:LinearGaussianStateSpaceModel},
    θ_cd::CoDual,
    ys_cd::CoDual{<:AbstractVector},
    controls_cd::CoDual{<:NamedTuple},
)
    filter = primal(filter_cd)
    model = primal(model_cd)
    θ = primal(θ_cd)
    ys = primal(ys_cd)
    controls = primal(controls_cd)

    hoisted_controls = hoist_controls(controls, θ)
    prior_hoist = hoist_static(prior(model), θ, hoisted_controls)
    dyn_hoist = hoist_static(dyn(model), θ, hoisted_controls)
    obs_hoist = hoist_static(obs(model), θ, hoisted_controls)

    prior_params = map(_val, step_params(prior(model), θ, hoisted_controls, prior_hoist))
    initial_state = _step_initial(filter, prior_params)

    T = length(ys)
    states = Vector{Any}(undef, T + 1)
    caches = Vector{Any}(undef, T)
    resolved_per_step = Vector{Any}(undef, T)
    states[1] = initial_state

    state = initial_state
    ll = zero(eltype(eltype(ys)))
    for t in 1:T
        resolved = resolve_controls(controls, hoisted_controls, θ, t)
        dyn_params = map(_val, step_params(dyn(model), θ, t, resolved, dyn_hoist))
        obs_params = map(_val, step_params(obs(model), θ, t, resolved, obs_hoist))
        new_state, ll_inc, cache = _step_forward(filter, state, dyn_params, obs_params, ys[t])
        states[t + 1] = new_state
        caches[t] = cache
        resolved_per_step[t] = resolved
        ll += ll_inc
        state = new_state
    end

    function ssm_loglikelihood_pb(Δll)
        ∂state = _zero_state_cotangent(filter, states[T + 1])
        ∂θ_rd = Mooncake.zero_rdata(θ)

        # Per-field FP accumulator buffers (Ref{Any} for lazy typing).
        dyn_bufs = (
            A=_make_fp_buffer(dyn(model).A),
            b=_make_fp_buffer(dyn(model).b),
            Q=_make_fp_buffer(dyn(model).Q),
        )
        obs_bufs = (
            H=_make_fp_buffer(obs(model).H),
            c=_make_fp_buffer(obs(model).c),
            R=_make_fp_buffer(obs(model).R),
        )
        prior_bufs = (
            μ0=_make_fp_buffer(prior(model).μ0),
            Σ0=_make_fp_buffer(prior(model).Σ0),
        )

        for t in T:-1:1
            ∂state, ∂dyn_p, ∂obs_p = _step_pullback(
                filter, ∂state, Δll, caches[t], dyn(model), obs(model)
            )
            resolved = resolved_per_step[t]
            ∂θ_rd = _route_dyn_step!(dyn_bufs, ∂θ_rd, dyn(model), ∂dyn_p, θ_cd, t, resolved)
            ∂θ_rd = _route_obs_step!(obs_bufs, ∂θ_rd, obs(model), ∂obs_p, θ_cd, t, resolved)
        end

        ∂prior_p = _initial_pullback(filter, ∂state, prior(model))
        ∂θ_rd = _route_prior!(prior_bufs, ∂θ_rd, prior(model), ∂prior_p, θ_cd, hoisted_controls)

        # End-of-loop pullbacks through FixedParametric closures.
        ∂θ_rd = _finish_dyn_fp!(∂θ_rd, dyn(model), dyn_bufs, θ_cd, hoisted_controls)
        ∂θ_rd = _finish_obs_fp!(∂θ_rd, obs(model), obs_bufs, θ_cd, hoisted_controls)
        ∂θ_rd = _finish_prior_fp!(∂θ_rd, prior(model), prior_bufs, θ_cd, hoisted_controls)

        return (NoRData(), NoRData(), NoRData(), ∂θ_rd, NoRData(), NoRData())
    end

    return CoDual(ll, NoFData()), ssm_loglikelihood_pb
end

end # module MooncakeExt
