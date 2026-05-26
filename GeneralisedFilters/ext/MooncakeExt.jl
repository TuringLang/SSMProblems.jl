"""
Mooncake.jl extension for GeneralisedFilters.

Provides a native Mooncake `rrule!!` for `_ssm_loglikelihood` (the positional helper
behind [`ssm_loglikelihood`](@ref)) on `LinearGaussianStateSpaceModel`.

Architecture:
- The forward pass mirrors `_ssm_loglikelihood`, caching per-step state caches and
  resolved controls so the backward sweep can reconstruct intermediates.
- The backward sweep walks the loop in reverse, calling [`_step_pullback`](@ref) to
  obtain per-step parameter cotangents. Each cotangent is routed by the corresponding
  model component field's trait:
    * `Fixed` / `TimeVarying`     -> no-op (the analytical math was elided already)
    * `FixedParametric`           -> accumulate into a buffer; one Mooncake pullback at
                                     end of loop through `p.f(θ, hoisted_controls)`
    * `TimeVaryingParametric`     -> per-step Mooncake pullback through `p.f(θ, t, resolved)`
- User-closure pullbacks are obtained via `Mooncake.build_rrule`. Each call uses a
  FRESH `θ_inner_cd = zero_fcodual(θ)` with the inner fdata explicitly accumulated into
  the outer `θ_cd.dx` afterwards — sharing one `θ_cd` across multiple inner rule calls
  corrupts the fdata when the user closure involves array slicing (see design doc).

Limitation: parametric parameters that read parametric controls (the "shared
θ-dependent computation" pattern) are not yet supported — `∂hoisted_controls` and
`∂resolved` contributions from parameter pullbacks are silently discarded.
"""
module MooncakeExt

using GeneralisedFilters:
    GeneralisedFilters,
    _ssm_loglikelihood,
    _step_forward,
    _step_pullback,
    _step_initial,
    _initial_pullback,
    Fixed,
    FixedParametric,
    TimeVarying,
    TimeVaryingParametric,
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

using Distributions: MvNormal, params
using PDMats: PDMat
using LinearAlgebra: Symmetric

using Mooncake: Mooncake, @is_primitive, CoDual, primal
using Mooncake: NoFData, NoRData, NoTangent, NoCache
using Mooncake: zero_tangent, zero_fcodual, rdata, fdata, primal_to_tangent!!
using Mooncake: increment_internal!!, build_rrule

## TANGENT DECLARATIONS ########################################################################

Mooncake.tangent_type(::Type{<:Fixed}) = NoTangent
Mooncake.tangent_type(::Type{<:TimeVarying}) = NoTangent
Mooncake.tangent_type(::Type{<:FixedParametric}) = NoTangent
Mooncake.tangent_type(::Type{<:TimeVaryingParametric}) = NoTangent

## HELPERS #####################################################################################

# Rdata accumulation: handle NoRData no-op explicitly; otherwise delegate.
_add_rdata(::NoRData, ::NoRData) = NoRData()
_add_rdata(::NoRData, b) = b
_add_rdata(a, ::NoRData) = a
_add_rdata(a, b) = increment_internal!!(NoCache(), a, b)

# Accumulate the inner CoDual's fdata into the outer's. NoFData (immutable types)
# is a no-op; for mutable types this is an in-place increment.
_accumulate_fdata!(::NoFData, ::NoFData) = nothing
_accumulate_fdata!(outer, inner) = (increment_internal!!(NoCache(), outer, inner); nothing)

# Zero-cotangent for state at output. We differentiate ll, not the final state.
function _zero_state_cotangent(::KalmanFilter, state::MvNormal)
    μ, Σ = params(state)
    Σ_inner = Σ isa PDMat ? Σ.mat : Matrix(Σ)
    return (zero(μ), zero(Σ_inner))
end

# FixedParametric per-field accumulator buffer. `Ref{Any}` keeps init lazy so we don't
# have to know the cotangent type up front (relevant for PDMat-typed params, whose
# cotangent type is a plain matrix, not a PDMat).
_make_fp_buffer(::Any) = nothing
_make_fp_buffer(::FixedParametric) = Base.RefValue{Any}(nothing)

_acc_fp!(::Nothing, _) = nothing
function _acc_fp!(buf::Base.RefValue{Any}, ∂val)
    buf[] = buf[] === nothing ? ∂val : buf[] + ∂val
    return nothing
end

# Per-step Mooncake pullback through a TimeVaryingParametric closure `f(θ, t, resolved)`.
# Uses a fresh inner θ CoDual per call (sharing the outer θ_cd's fdata across multiple
# rule invocations triggers a Mooncake misbehaviour for closures involving array
# slicing — each call doubles existing fdata). After the pullback, the inner fdata is
# accumulated into the outer θ_cd's fdata explicitly.
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

## ROUTING #####################################################################################

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

function _route_dyn_step!(
    bufs, ∂θ_rd, dyn::LinearGaussianLatentDynamics, ∂vals, θ_cd, t, resolved
)
    ∂θ_rd = _route_param!(bufs.A, ∂θ_rd, dyn.A, ∂vals.A, θ_cd, t, resolved)
    ∂θ_rd = _route_param!(bufs.b, ∂θ_rd, dyn.b, ∂vals.b, θ_cd, t, resolved)
    ∂θ_rd = _route_param!(bufs.Q, ∂θ_rd, dyn.Q, ∂vals.Q, θ_cd, t, resolved)
    return ∂θ_rd
end

function _route_obs_step!(
    bufs, ∂θ_rd, obs::LinearGaussianObservationProcess, ∂vals, θ_cd, t, resolved
)
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
    ∂θ_rd = _route_prior_field!(
        bufs.μ0, ∂θ_rd, prior.μ0, ∂vals.μ0, θ_cd, hoisted_controls
    )
    ∂θ_rd = _route_prior_field!(
        bufs.Σ0, ∂θ_rd, prior.Σ0, ∂vals.Σ0, θ_cd, hoisted_controls
    )
    return ∂θ_rd
end

# End-of-loop FixedParametric pullback per field.
_finish_fp_field!(∂θ_rd, _, ::Nothing, _, _) = ∂θ_rd
function _finish_fp_field!(
    ∂θ_rd, p::FixedParametric, buf::Base.RefValue{Any}, θ_cd, hoisted_controls
)
    buf[] === nothing && return ∂θ_rd
    return _add_rdata(∂θ_rd, _fp_finish_pullback!(p.f, buf[], θ_cd, hoisted_controls))
end

function _finish_dyn_fp!(
    ∂θ_rd, dyn::LinearGaussianLatentDynamics, bufs, θ_cd, hoisted_controls
)
    ∂θ_rd = _finish_fp_field!(∂θ_rd, dyn.A, bufs.A, θ_cd, hoisted_controls)
    ∂θ_rd = _finish_fp_field!(∂θ_rd, dyn.b, bufs.b, θ_cd, hoisted_controls)
    ∂θ_rd = _finish_fp_field!(∂θ_rd, dyn.Q, bufs.Q, θ_cd, hoisted_controls)
    return ∂θ_rd
end

function _finish_obs_fp!(
    ∂θ_rd, obs::LinearGaussianObservationProcess, bufs, θ_cd, hoisted_controls
)
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

## RRULE!! #####################################################################################

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

    prior_params = step_params(prior(model), θ, hoisted_controls, prior_hoist)
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
        dyn_params = step_params(dyn(model), θ, t, resolved, dyn_hoist)
        obs_params = step_params(obs(model), θ, t, resolved, obs_hoist)
        new_state, ll_inc, cache = _step_forward(
            filter, state, dyn_params, obs_params, ys[t]
        )
        states[t + 1] = new_state
        caches[t] = cache
        resolved_per_step[t] = resolved
        ll += ll_inc
        state = new_state
    end

    function ssm_loglikelihood_pb(Δll)
        ∂state = _zero_state_cotangent(filter, states[T + 1])
        ∂θ_rd = Mooncake.zero_rdata(θ)

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
            ∂θ_rd = _route_dyn_step!(
                dyn_bufs, ∂θ_rd, dyn(model), ∂dyn_p, θ_cd, t, resolved
            )
            ∂θ_rd = _route_obs_step!(
                obs_bufs, ∂θ_rd, obs(model), ∂obs_p, θ_cd, t, resolved
            )
        end

        ∂prior_p = _initial_pullback(filter, ∂state, prior(model))
        ∂θ_rd = _route_prior!(
            prior_bufs, ∂θ_rd, prior(model), ∂prior_p, θ_cd, hoisted_controls
        )

        ∂θ_rd = _finish_dyn_fp!(∂θ_rd, dyn(model), dyn_bufs, θ_cd, hoisted_controls)
        ∂θ_rd = _finish_obs_fp!(∂θ_rd, obs(model), obs_bufs, θ_cd, hoisted_controls)
        ∂θ_rd = _finish_prior_fp!(∂θ_rd, prior(model), prior_bufs, θ_cd, hoisted_controls)

        return (NoRData(), NoRData(), NoRData(), ∂θ_rd, NoRData(), NoRData())
    end

    return CoDual(ll, NoFData()), ssm_loglikelihood_pb
end

end # module MooncakeExt
