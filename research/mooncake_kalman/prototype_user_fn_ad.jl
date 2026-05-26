## Prototype validating: a custom Mooncake rrule!! on `outer(θ, a, b, c)` that
## internally needs gradients of user-supplied closures `a.f(θ)`, `b.f(θ)`, while
## skipping `c::Fixed` at compile time. The pattern mimics what `ssm_loglikelihood`'s
## rrule will need for FixedParametric / TimeVaryingParametric parameters.

using Mooncake
using FiniteDifferences
using LinearAlgebra
using StaticArrays

const M = Mooncake

## Trait wrappers (matching the real package)

struct Fixed{T}
    value::T
end

struct FixedParametric{F}
    f::F
end

M.tangent_type(::Type{<:Fixed}) = M.NoTangent
M.tangent_type(::Type{<:FixedParametric}) = M.NoTangent

## The function under test

_val(p::Fixed, _) = p.value
_val(p::FixedParametric, θ) = p.f(θ)

function outer(θ, a, b, c)
    val_a = _val(a, θ)
    val_b = _val(b, θ)
    val_c = _val(c, θ)
    return sum(val_a .* val_b) + sum(val_c)
end

## Helpers
##
## Strategy: pass the outer θ_cd directly to each inner rule call. Inner pullbacks
## that mutate fdata accumulate into our θ_cd's fdata buffer automatically. For
## immutable θ (e.g. SVector), the inner pb returns rdata which we accumulate
## manually into an rdata accumulator built from `zero_tangent`.

_param_grad!(rdata_acc, ::Fixed, _, _) = rdata_acc

function _param_grad!(rdata_acc, p::FixedParametric, ∂val, θ_cd::M.CoDual)
    θ = M.primal(θ_cd)
    rule = M.build_rrule(Tuple{typeof(p.f),typeof(θ)})
    out_cd, pb = rule(M.zero_fcodual(p.f), θ_cd)
    out_primal = M.primal(out_cd)
    # Split primal-shaped ∂val into fdata + rdata. For mutable outputs (Vector),
    # the fdata accumulates into out_cd.dx; rdata is passed to pb.
    ∂val_tangent = M.primal_to_tangent!!(M.zero_tangent(out_primal), ∂val)
    ∂val_fdata = M.fdata(∂val_tangent)
    ∂val_rdata = M.rdata(∂val_tangent)
    M.increment_internal!!(M.NoCache(), out_cd.dx, ∂val_fdata)
    _, ∂θ_rdata = pb(∂val_rdata)
    return _add_rdata(rdata_acc, ∂θ_rdata)
end

# Add two rdata values. Handles NoRData + NoRData (no-op) and proper rdata addition.
_add_rdata(::M.NoRData, ::M.NoRData) = M.NoRData()
_add_rdata(a, b) = M.increment_internal!!(M.NoCache(), a, b)

## rrule!!

function M.rrule!!(
    ::M.CoDual{typeof(outer)},
    θ_cd::M.CoDual,
    a_cd::M.CoDual,
    b_cd::M.CoDual,
    c_cd::M.CoDual,
)
    θ = M.primal(θ_cd)
    a = M.primal(a_cd)
    b = M.primal(b_cd)
    c = M.primal(c_cd)

    val_a = _val(a, θ)
    val_b = _val(b, θ)
    val_c = _val(c, θ)

    result = sum(val_a .* val_b) + sum(val_c)

    function outer_pb(Δresult)
        ∂val_a = Δresult .* val_b
        ∂val_b = Δresult .* val_a

        # Start from zero rdata for θ. Mutable types: NoRData. Immutable types: zero rdata.
        rdata_acc = M.zero_rdata(θ)
        rdata_acc = _param_grad!(rdata_acc, a, ∂val_a, θ_cd)
        rdata_acc = _param_grad!(rdata_acc, b, ∂val_b, θ_cd)
        # c is Fixed — contribution elided

        return (M.NoRData(), rdata_acc, M.NoRData(), M.NoRData(), M.NoRData())
    end

    return M.CoDual(result, M.NoFData()), outer_pb
end

@M.is_primitive M.DefaultCtx Tuple{typeof(outer),Any,Any,Any,Any}

## Test

const Dx = 3

function run_test(θ_test, label; shared=false)
    if shared
        # Closures share a common helper that depends on θ — exercises the fan-in
        # case where two parametric pullbacks both contribute to ∂θ via a shared
        # subexpression. Mooncake auto-AD should handle the accumulation.
        shared_helper(θ) = exp.(θ ./ 2)
        a = FixedParametric(θ -> shared_helper(θ) .* θ)         # uses θ twice
        b = FixedParametric(θ -> shared_helper(θ) .+ θ.^2)      # uses θ twice
    else
        a = FixedParametric(θ -> θ .^ 2)
        b = FixedParametric(θ -> sin.(θ))
    end
    c = Fixed(SVector(1.0, 2.0, 3.0))

    f(θ) = outer(θ, a, b, c)

    fdm = central_fdm(5, 1)
    grad_fd = FiniteDifferences.grad(fdm, f, collect(θ_test))[1]

    cache = M.prepare_gradient_cache(f, θ_test)
    _, (_, grad_mc) = M.value_and_gradient!!(cache, f, θ_test)

    grad_mc_vec = _to_vec(grad_mc)
    println("--- $label ---")
    println("grad_fd = $(collect(grad_fd))")
    println("grad_mc = $(grad_mc_vec)")
    println("match   = $(isapprox(grad_mc_vec, collect(grad_fd); rtol=1e-5))")
    return nothing
end

_to_vec(x::AbstractVector) = collect(x)
_to_vec(t::M.Tangent) = collect(t.fields.data)

run_test(randn(Dx), "mutable Vector{Float64}")
run_test((@SVector randn(Dx)), "immutable SVector")
run_test(randn(Dx), "mutable Vector{Float64} (shared)"; shared=true)
run_test((@SVector randn(Dx)), "immutable SVector (shared)"; shared=true)
