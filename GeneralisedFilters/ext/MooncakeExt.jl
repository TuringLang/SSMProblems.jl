"""
Mooncake.jl extension for GeneralisedFilters.

Provides a native Mooncake `rrule!!` for `_ssm_loglikelihood` (the positional helper
behind [`ssm_loglikelihood`](@ref)). Dispatches on any `AbstractFilter` whose step
interface is implemented (`_step_initial` / `_step_forward` / `_step_pullback` /
`_initial_pullback` / `_zero_state_cotangent`) and any model matching
`ParameterisedSSM` (currently `LinearGaussianStateSpaceModel`).

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
- User-closure pullbacks are obtained via `Mooncake.build_rrule`. For per-step (TVP)
  closures the rule is built ONCE at the start of the backward sweep and reused across
  every step (avoiding Mooncake's per-call `_copy` of the rule captures — the dominant
  per-step cost). Each call still uses a FRESH `θ_inner_cd = zero_fcodual(θ)` with the
  inner fdata explicitly accumulated into the outer `θ_cd.dx` afterwards — sharing one
  `θ_cd` across inner rule calls corrupts the fdata when the user closure involves array
  slicing (see design doc). Reusing the rule object is safe precisely because the inputs
  are fresh; only the rule's internal captures are reused, and each forward overwrites
  them before its pullback reads them.
- Parametric controls are supported. Per step, ∂resolved contributions from all TVP
  parameters are summed and routed by each control's trait: TVP controls fire a per-step
  Mooncake pullback through `p.f(θ, t)`; FP controls accumulate into a control-side
  buffer. End-of-loop, ∂hoisted contributions from FP parameters route into the same
  FP-control buffers, then each FP control fires one Mooncake pullback through `p.f(θ)`.
"""
module MooncakeExt

using GeneralisedFilters:
    GeneralisedFilters,
    _ssm_loglikelihood,
    _step_forward,
    _step_pullback,
    _step_initial,
    _initial_pullback,
    _zero_state_cotangent,
    Fixed,
    FixedParametric,
    TimeVarying,
    TimeVaryingParametric,
    hoist_controls,
    hoist_static,
    resolve_controls,
    step_params,
    ParameterisedSSM
using SSMProblems: prior, dyn, obs

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

# Per-field routing state, built once at the start of the backward sweep and reused
# across all T steps:
#   - FixedParametric        -> an accumulator buffer (`Ref{Any}`) for the end-of-loop
#                               pullback. `Ref{Any}` keeps init lazy so we don't need the
#                               cotangent type up front (a PDMat param's cotangent is a
#                               plain matrix, not a PDMat).
#   - TimeVaryingParametric  -> a prebuilt Mooncake rule, reused across every step.
#   - Fixed / TimeVarying    -> nothing.
#
# Reusing one rule across steps is safe because each step passes FRESH input CoDuals
# (`θ_inner`, `resolved_inner`); only the rule's internal capture buffers are reused, and
# those are fully overwritten by each forward pass before the matching pullback reads
# them. (The original slicing bug was about sharing a non-zero *input* CoDual, not the
# rule object.) Building the rule once instead of per-step avoids Mooncake's per-call
# `_copy` of the rule's captures — the dominant cost of the per-step pullback.
_make_route_state(::Union{Fixed,TimeVarying}, θ, t, resolved) = nothing
_make_route_state(::FixedParametric, θ, t, resolved) = Base.RefValue{Any}(nothing)
function _make_route_state(p::TimeVaryingParametric, θ, t, resolved)
    return build_rrule(Tuple{typeof(p.f),typeof(θ),typeof(t),typeof(resolved)})
end

# Control routing state. TVP control signature is `f(θ, t)` (no `resolved`).
_make_ctrl_route_state(::Union{Fixed,TimeVarying}, θ, t) = nothing
_make_ctrl_route_state(::FixedParametric, θ, t) = Base.RefValue{Any}(nothing)
function _make_ctrl_route_state(p::TimeVaryingParametric, θ, t)
    return build_rrule(Tuple{typeof(p.f),typeof(θ),typeof(t)})
end

# Prior accumulator buffer (priors are only Fixed / FixedParametric — no rules needed).
_make_fp_buffer(::Any) = nothing
_make_fp_buffer(::FixedParametric) = Base.RefValue{Any}(nothing)

_acc_fp!(::Nothing, _) = nothing
function _acc_fp!(buf::Base.RefValue{Any}, ∂val)
    buf[] = buf[] === nothing ? ∂val : buf[] + ∂val
    return nothing
end

# Per-step Mooncake pullback through a TimeVaryingParametric parameter closure
# `f(θ, t, resolved) -> param_value`, using the prebuilt reusable `rule`. Uses a fresh
# inner θ CoDual per call (sharing the outer θ_cd's fdata across multiple rule
# invocations triggers a Mooncake misbehaviour for closures involving array slicing).
# After the pullback, the inner fdata is accumulated into the outer θ_cd's fdata.
#
# Returns `(∂θ_rd, ∂resolved_tan)` where `∂resolved_tan` is a NamedTuple of per-field
# tangents on `resolved` (used for routing parametric-controls contributions).
function _tvp_step_pullback!(rule, f, ∂val, θ_cd, t, resolved)
    θ = primal(θ_cd)
    θ_inner = zero_fcodual(θ)
    resolved_inner = zero_fcodual(resolved)
    out_cd, pb = rule(zero_fcodual(f), θ_inner, zero_fcodual(t), resolved_inner)
    out_primal = primal(out_cd)
    ∂val_tan = primal_to_tangent!!(zero_tangent(out_primal), ∂val)
    increment_internal!!(NoCache(), out_cd.dx, fdata(∂val_tan))
    _, ∂θ_rd, _, ∂resolved_rd = pb(rdata(∂val_tan))
    _accumulate_fdata!(θ_cd.dx, θ_inner.dx)
    ∂resolved_tan = Mooncake.tangent(resolved_inner.dx, ∂resolved_rd)
    return ∂θ_rd, ∂resolved_tan
end

# Per-step Mooncake pullback through a TimeVaryingParametric *control* closure
# `f(θ, t) -> control_value`, using the prebuilt reusable `rule`. Returns only ∂θ_rd.
function _tvp_control_step_pullback!(rule, f, ∂val, θ_cd, t)
    θ = primal(θ_cd)
    θ_inner = zero_fcodual(θ)
    out_cd, pb = rule(zero_fcodual(f), θ_inner, zero_fcodual(t))
    out_primal = primal(out_cd)
    ∂val_tan = primal_to_tangent!!(zero_tangent(out_primal), ∂val)
    increment_internal!!(NoCache(), out_cd.dx, fdata(∂val_tan))
    _, ∂θ_rd, _ = pb(rdata(∂val_tan))
    _accumulate_fdata!(θ_cd.dx, θ_inner.dx)
    return ∂θ_rd
end

# End-of-loop Mooncake pullback through a FixedParametric parameter closure
# `f(θ, hoisted_controls) -> param_value`. Returns `(∂θ_rd, ∂hoisted_tan)`.
function _fp_finish_pullback!(f, ∂val, θ_cd, hoisted_controls)
    θ = primal(θ_cd)
    rule = build_rrule(Tuple{typeof(f),typeof(θ),typeof(hoisted_controls)})
    θ_inner = zero_fcodual(θ)
    hoisted_inner = zero_fcodual(hoisted_controls)
    out_cd, pb = rule(zero_fcodual(f), θ_inner, hoisted_inner)
    out_primal = primal(out_cd)
    ∂val_tan = primal_to_tangent!!(zero_tangent(out_primal), ∂val)
    increment_internal!!(NoCache(), out_cd.dx, fdata(∂val_tan))
    _, ∂θ_rd, ∂hoisted_rd = pb(rdata(∂val_tan))
    _accumulate_fdata!(θ_cd.dx, θ_inner.dx)
    ∂hoisted_tan = Mooncake.tangent(hoisted_inner.dx, ∂hoisted_rd)
    return ∂θ_rd, ∂hoisted_tan
end

# End-of-loop Mooncake pullback through a FixedParametric *control* closure
# `f(θ) -> control_value`. Returns only ∂θ_rd.
function _fp_control_finish_pullback!(f, ∂val, θ_cd)
    θ = primal(θ_cd)
    rule = build_rrule(Tuple{typeof(f),typeof(θ)})
    θ_inner = zero_fcodual(θ)
    out_cd, pb = rule(zero_fcodual(f), θ_inner)
    out_primal = primal(out_cd)
    ∂val_tan = primal_to_tangent!!(zero_tangent(out_primal), ∂val)
    increment_internal!!(NoCache(), out_cd.dx, fdata(∂val_tan))
    _, ∂θ_rd = pb(rdata(∂val_tan))
    _accumulate_fdata!(θ_cd.dx, θ_inner.dx)
    return ∂θ_rd
end

## ROUTING #####################################################################################
#
# Component walkers are generic over the component's NamedTuple-of-parameter-fields layout.
# `@generated` unrolls the field loop at compile time so the dispatch on each wrapper's
# trait (`_route_param!` / `_route_param_prior!`) stays type-stable per concrete component
# type. Adding a new model component (any struct of `AbstractModelParameter` fields)
# requires no MooncakeExt changes.

# Per-field step trait dispatch (used by dyn / obs). Returns `(∂θ_rd, ∂resolved_tan)`
# where ∂resolved_tan is `nothing` for non-TVP fields and a NamedTuple of per-control
# tangents for TVP fields (used to route parametric-controls contributions).
_route_param!(_, ∂θ_rd, ::Union{Fixed,TimeVarying}, _, _, _, _) = (∂θ_rd, nothing)
function _route_param!(buf, ∂θ_rd, ::FixedParametric, ∂val, _, _, _)
    _acc_fp!(buf, ∂val)
    return (∂θ_rd, nothing)
end
function _route_param!(rule, ∂θ_rd, p::TimeVaryingParametric, ∂val, θ_cd, t, resolved)
    ∂θ_p, ∂resolved_tan = _tvp_step_pullback!(rule, p.f, ∂val, θ_cd, t, resolved)
    return (_add_rdata(∂θ_rd, ∂θ_p), ∂resolved_tan)
end

# Per-field prior trait dispatch (no time index; TVP forbidden).
_route_param_prior!(_, ∂θ_rd, ::Union{Fixed,TimeVarying}, _) = ∂θ_rd
function _route_param_prior!(buf, ∂θ_rd, ::FixedParametric, ∂val)
    _acc_fp!(buf, ∂val)
    return ∂θ_rd
end
function _route_param_prior!(_, _, ::TimeVaryingParametric, _)
    return error(
        "TimeVaryingParametric is not valid for prior parameters; use FixedParametric"
    )
end

# End-of-loop FixedParametric pullback per field. Returns `(∂θ_rd, ∂hoisted_tan)` where
# ∂hoisted_tan is `nothing` when there's no FP contribution (buf empty), else a
# NamedTuple of per-control tangents into `hoisted_controls`. The default no-ops for
# non-FP fields (their route state is `nothing` or a TVP rule, not an FP buffer).
_finish_fp_field!(∂θ_rd, _, _, _, _) = (∂θ_rd, nothing)
function _finish_fp_field!(
    ∂θ_rd, p::FixedParametric, buf::Base.RefValue{Any}, θ_cd, hoisted_controls
)
    buf[] === nothing && return (∂θ_rd, nothing)
    ∂θ_p, ∂hoisted_tan = _fp_finish_pullback!(p.f, buf[], θ_cd, hoisted_controls)
    return (_add_rdata(∂θ_rd, ∂θ_p), ∂hoisted_tan)
end

# Per-control step trait dispatch. The `∂val` is the cotangent on the resolved control
# value for one step.
_route_control!(_, ∂θ_rd, ::Union{Fixed,TimeVarying}, _, _, _) = ∂θ_rd
function _route_control!(buf, ∂θ_rd, ::FixedParametric, ∂val, _, _)
    _acc_fp!(buf, ∂val)
    return ∂θ_rd
end
function _route_control!(rule, ∂θ_rd, p::TimeVaryingParametric, ∂val, θ_cd, t)
    return _add_rdata(∂θ_rd, _tvp_control_step_pullback!(rule, p.f, ∂val, θ_cd, t))
end

# Per-control end-of-loop FP-control trait dispatch. The `∂val` is the cotangent on
# the (Fixed or FP) entry of `hoisted_controls`. TVP controls don't appear in
# `hoisted_controls` (they're nothing in hoist), so their hoist-cotangent path is
# never exercised.
_route_hoisted_control!(_, ∂θ_rd, ::Union{Fixed,TimeVarying}, _) = ∂θ_rd
function _route_hoisted_control!(buf, ∂θ_rd, ::FixedParametric, ∂val)
    _acc_fp!(buf, ∂val)
    return ∂θ_rd
end
function _route_hoisted_control!(_, ∂θ_rd, ::TimeVaryingParametric, _)
    # Shouldn't fire — TVP entries in hoisted_controls are `nothing` and shouldn't
    # carry a contributing cotangent. Defensive no-op rather than erroring.
    return ∂θ_rd
end

# End-of-loop FP-control pullback per field. ∂val is the buffer's accumulated tangent.
# The default no-ops for non-FP controls (route state is `nothing` or a TVP rule).
_finish_fp_control_field!(∂θ_rd, _, _, _) = ∂θ_rd
function _finish_fp_control_field!(
    ∂θ_rd, p::FixedParametric, buf::Base.RefValue{Any}, θ_cd
)
    buf[] === nothing && return ∂θ_rd
    return _add_rdata(∂θ_rd, _fp_control_finish_pullback!(p.f, buf[], θ_cd))
end

# Per-component route-state NamedTuple (FP buffers + prebuilt TVP rules), keyed by the
# component's field names. `t` / `resolved` are sample values used only to form the TVP
# rule signature.
@generated function _make_route_states(component, θ, t, resolved)
    pairs = [
        Expr(
            :(=),
            k,
            :(_make_route_state(getfield(component, $(QuoteNode(k))), θ, t, resolved)),
        ) for k in fieldnames(component)
    ]
    return Expr(:tuple, pairs...)
end

# Per-controls route-state NamedTuple (FP buffers + prebuilt TVP control rules).
@generated function _make_ctrl_route_states(controls::NamedTuple{N}, θ, t) where {N}
    pairs = [
        Expr(:(=), k, :(_make_ctrl_route_state(getfield(controls, $(QuoteNode(k))), θ, t)))
        for k in N
    ]
    return Expr(:tuple, pairs...)
end

# Prior accumulator NamedTuple (FP buffers only; priors have no TVP fields).
@generated function _make_fp_buffers(component)
    pairs = [
        Expr(:(=), k, :(_make_fp_buffer(getfield(component, $(QuoteNode(k)))))) for
        k in fieldnames(component)
    ]
    return Expr(:tuple, pairs...)
end

# Step routing: walk a component's fields, dispatching each via _route_param!.
# Returns `(∂θ_rd, ∂resolved_acc)` where ∂resolved_acc is `nothing` (no TVP fields in
# this component) or a NamedTuple of per-control tangents summed across this component's
# TVP fields.
@generated function _route_component_step!(
    bufs::NamedTuple{N}, ∂θ_rd, component, ∂vals::NamedTuple{N}, θ_cd, t, resolved
) where {N}
    exprs = Expr[:(∂resolved_acc = nothing)]
    for k in N
        sym = QuoteNode(k)
        push!(
            exprs,
            :(begin
                ∂θ_rd, ∂res = _route_param!(
                    getfield(bufs, $sym),
                    ∂θ_rd,
                    getfield(component, $sym),
                    getfield(∂vals, $sym),
                    θ_cd,
                    t,
                    resolved,
                )
                ∂resolved_acc = _sum_nt(∂resolved_acc, ∂res)
            end),
        )
    end
    return Expr(:block, exprs..., :(return (∂θ_rd, ∂resolved_acc)))
end

# Prior routing: walk a prior's fields, dispatching each via _route_param_prior!.
@generated function _route_prior_component!(
    bufs::NamedTuple{N}, ∂θ_rd, prior, ∂vals::NamedTuple{N}
) where {N}
    exprs = Expr[]
    for k in N
        sym = QuoteNode(k)
        push!(
            exprs,
            :(∂θ_rd = _route_param_prior!(
                getfield(bufs, $sym),
                ∂θ_rd,
                getfield(prior, $sym),
                getfield(∂vals, $sym),
            )),
        )
    end
    return Expr(:block, exprs..., :(return ∂θ_rd))
end

# End-of-loop FP finishing for one component. Returns `(∂θ_rd, ∂hoisted_acc)` where
# ∂hoisted_acc accumulates per-control tangents into hoisted_controls across all FP
# fields of this component.
@generated function _finish_fp_component!(
    ∂θ_rd, component, bufs::NamedTuple{N}, θ_cd, hoisted_controls
) where {N}
    exprs = Expr[:(∂hoisted_acc = nothing)]
    for k in N
        sym = QuoteNode(k)
        push!(
            exprs,
            :(begin
                ∂θ_rd, ∂hoist = _finish_fp_field!(
                    ∂θ_rd,
                    getfield(component, $sym),
                    getfield(bufs, $sym),
                    θ_cd,
                    hoisted_controls,
                )
                ∂hoisted_acc = _sum_nt(∂hoisted_acc, ∂hoist)
            end),
        )
    end
    return Expr(:block, exprs..., :(return (∂θ_rd, ∂hoisted_acc)))
end

## CONTROL ROUTING #############################################################################

# Field-wise NamedTuple sum that tolerates either side being `nothing`. Used to
# accumulate ∂resolved_step and ∂hoisted_acc across components.
_sum_nt(a, ::Nothing) = a
_sum_nt(::Nothing, b) = b
_sum_nt(::Nothing, ::Nothing) = nothing
@inline _sum_nt(a::NamedTuple{N}, b::NamedTuple{N}) where {N} = map(_sum_field, a, b)
# Field-level helper: drop NoTangent or otherwise-absent contributions.
_sum_field(a, ::Mooncake.NoTangent) = a
_sum_field(::Mooncake.NoTangent, b) = b
_sum_field(::Mooncake.NoTangent, ::Mooncake.NoTangent) = Mooncake.NoTangent()
_sum_field(a, b) = a + b

# Per-step controls routing: walk the controls NamedTuple, dispatching each on the
# control's trait. `∂resolved_tan` is a NamedTuple with the same keys as `controls`.
@generated function _route_controls_step!(
    ctrl_states::NamedTuple{N},
    ∂θ_rd,
    controls::NamedTuple{N},
    ∂resolved_tan::NamedTuple{N},
    θ_cd,
    t,
) where {N}
    exprs = Expr[]
    for k in N
        sym = QuoteNode(k)
        push!(
            exprs,
            :(∂θ_rd = _route_control!(
                getfield(ctrl_states, $sym),
                ∂θ_rd,
                getfield(controls, $sym),
                getfield(∂resolved_tan, $sym),
                θ_cd,
                t,
            )),
        )
    end
    return Expr(:block, exprs..., :(return ∂θ_rd))
end

# End-of-loop controls routing for the ∂hoisted accumulator. Fixed/FP entries route as
# before; TVP entries are no-ops because TVP controls aren't in `hoisted_controls`.
@generated function _route_hoisted_controls!(
    ctrl_states::NamedTuple{N},
    ∂θ_rd,
    controls::NamedTuple{N},
    ∂hoisted_tan::NamedTuple{N},
) where {N}
    exprs = Expr[]
    for k in N
        sym = QuoteNode(k)
        push!(
            exprs,
            :(∂θ_rd = _route_hoisted_control!(
                getfield(ctrl_states, $sym),
                ∂θ_rd,
                getfield(controls, $sym),
                getfield(∂hoisted_tan, $sym),
            )),
        )
    end
    return Expr(:block, exprs..., :(return ∂θ_rd))
end

# Fire FP-control pullbacks at end of loop.
@generated function _finish_fp_controls!(
    ∂θ_rd, controls::NamedTuple{N}, ctrl_states::NamedTuple{N}, θ_cd
) where {N}
    exprs = Expr[]
    for k in N
        sym = QuoteNode(k)
        push!(
            exprs,
            :(∂θ_rd = _finish_fp_control_field!(
                ∂θ_rd, getfield(controls, $sym), getfield(ctrl_states, $sym), θ_cd
            )),
        )
    end
    return Expr(:block, exprs..., :(return ∂θ_rd))
end

## RRULE!! #####################################################################################

@is_primitive Mooncake.DefaultCtx Tuple{
    typeof(_ssm_loglikelihood),
    GeneralisedFilters.AbstractFilter,
    ParameterisedSSM,
    Any,
    AbstractVector,
    NamedTuple,
}

function Mooncake.rrule!!(
    ::CoDual{typeof(_ssm_loglikelihood)},
    filter_cd::CoDual{<:GeneralisedFilters.AbstractFilter},
    model_cd::CoDual{<:ParameterisedSSM},
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

    # Peel the first step to establish concrete cache / resolved types, so the storage
    # vectors below are type-stable (the per-step types are homogeneous across t). Only
    # the caches and resolved controls are needed by the backward sweep; intermediate
    # states are not (the caches carry everything), so we keep only the final state.
    resolved_1 = resolve_controls(controls, hoisted_controls, θ, 1)
    dyn_params_1 = step_params(dyn(model), θ, 1, resolved_1, dyn_hoist)
    obs_params_1 = step_params(obs(model), θ, 1, resolved_1, obs_hoist)
    state, ll, cache_1 = _step_forward(
        filter, initial_state, dyn_params_1, obs_params_1, ys[1]
    )

    caches = Vector{typeof(cache_1)}(undef, T)
    resolved_per_step = Vector{typeof(resolved_1)}(undef, T)
    caches[1] = cache_1
    resolved_per_step[1] = resolved_1

    for t in 2:T
        resolved = resolve_controls(controls, hoisted_controls, θ, t)
        dyn_params = step_params(dyn(model), θ, t, resolved, dyn_hoist)
        obs_params = step_params(obs(model), θ, t, resolved, obs_hoist)
        state, ll_inc, cache = _step_forward(
            filter, state, dyn_params, obs_params, ys[t]
        )
        caches[t] = cache
        resolved_per_step[t] = resolved
        ll += ll_inc
    end
    final_state = state

    function ssm_loglikelihood_pb(Δll)
        ∂state = _zero_state_cotangent(filter, final_state)
        ∂θ_rd = Mooncake.zero_rdata(θ)

        # Route states (FP buffers + prebuilt-once TVP rules), reused across all steps.
        # The sample t / resolved (step 1) only set the TVP rule signatures.
        resolved_1 = resolved_per_step[1]
        dyn_states = _make_route_states(dyn(model), θ, 1, resolved_1)
        obs_states = _make_route_states(obs(model), θ, 1, resolved_1)
        prior_bufs = _make_fp_buffers(prior(model))
        ctrl_states = _make_ctrl_route_states(controls, θ, 1)

        for t in T:-1:1
            ∂state, ∂dyn_p, ∂obs_p = _step_pullback(
                filter, ∂state, Δll, caches[t], dyn(model), obs(model)
            )
            resolved = resolved_per_step[t]
            ∂θ_rd, ∂res_dyn = _route_component_step!(
                dyn_states, ∂θ_rd, dyn(model), ∂dyn_p, θ_cd, t, resolved
            )
            ∂θ_rd, ∂res_obs = _route_component_step!(
                obs_states, ∂θ_rd, obs(model), ∂obs_p, θ_cd, t, resolved
            )
            ∂resolved_step = _sum_nt(∂res_dyn, ∂res_obs)
            if ∂resolved_step !== nothing
                ∂θ_rd = _route_controls_step!(
                    ctrl_states, ∂θ_rd, controls, ∂resolved_step, θ_cd, t
                )
            end
        end

        ∂prior_p = _initial_pullback(filter, ∂state, prior(model))
        ∂θ_rd = _route_prior_component!(prior_bufs, ∂θ_rd, prior(model), ∂prior_p)

        ∂θ_rd, ∂hoist_dyn = _finish_fp_component!(
            ∂θ_rd, dyn(model), dyn_states, θ_cd, hoisted_controls
        )
        ∂θ_rd, ∂hoist_obs = _finish_fp_component!(
            ∂θ_rd, obs(model), obs_states, θ_cd, hoisted_controls
        )
        ∂θ_rd, ∂hoist_prior = _finish_fp_component!(
            ∂θ_rd, prior(model), prior_bufs, θ_cd, hoisted_controls
        )
        ∂hoisted_acc = _sum_nt(_sum_nt(∂hoist_dyn, ∂hoist_obs), ∂hoist_prior)
        if ∂hoisted_acc !== nothing
            ∂θ_rd = _route_hoisted_controls!(
                ctrl_states, ∂θ_rd, controls, ∂hoisted_acc
            )
        end

        ∂θ_rd = _finish_fp_controls!(∂θ_rd, controls, ctrl_states, θ_cd)

        return (NoRData(), NoRData(), NoRData(), ∂θ_rd, NoRData(), NoRData())
    end

    return CoDual(ll, NoFData()), ssm_loglikelihood_pb
end

end # module MooncakeExt
