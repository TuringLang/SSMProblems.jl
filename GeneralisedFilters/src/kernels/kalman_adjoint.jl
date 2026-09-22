# Engine-agnostic analytic reverse pass for one fused Kalman step. It consumes the cache
# produced by `kalman_step_cached` and the output cotangents `(Δμ, ΔΣ, δll)` and returns the
# cotangents of the initial state and the dynamics/observation atoms. The reverse core never
# reads the filtered covariance, so the Joseph-form primal and the plain-form adjoint below
# describe the same mathematical function.

"Shared reverse recursion propagating the filtered-state cotangent backwards."
function _kalman_reverse_core(c, Δμ, ΔΣ, δll)
    # update reverse
    v̄ = -δll * c.w + c.K' * Δμ
    ŷ̄ = -v̄
    K̄ = -ΔΣ * c.Σ̂ * c.H' + Δμ * c.v'
    Sī = c.H * c.Σ̂ * K̄
    S̄ = δll * (c.w * c.w' - c.Si) / 2 - c.Si * Sī * c.Si
    μ̂̄ = Δμ + c.H' * ŷ̄
    Σ̂̄ = ΔΣ - c.H' * c.K' * ΔΣ + K̄ * c.Si * c.H + c.H' * S̄ * c.H
    # predict reverse
    μ0̄ = c.A' * μ̂̄
    Σ0̄ = c.A' * Σ̂̄ * c.A
    return (; μ0̄, Σ0̄, μ̂̄, Σ̂̄, ŷ̄, K̄, S̄, ΔΣ)
end

_A_adjoint(c, g) = g.μ̂̄ * c.μ0' + (g.Σ̂̄ + g.Σ̂̄') * c.A * c.Σ0
_A_adjoint(::Val{true}, c, g) = _A_adjoint(c, g)
_A_adjoint(::Val{false}, c, g) = zero(c.A)

_b_adjoint(c, g) = g.μ̂̄
_b_adjoint(::Val{true}, c, g) = _b_adjoint(c, g)
_b_adjoint(::Val{false}, c, g) = zero(g.μ̂̄)

_Q_adjoint(c, g) = g.Σ̂̄
_Q_adjoint(::Val{true}, c, g) = _Q_adjoint(c, g)
_Q_adjoint(::Val{false}, c, g) = zero(g.Σ̂̄)

function _H_adjoint(c, g)
    return -c.K' * g.ΔΣ * c.Σ̂ + c.Si * g.K̄' * c.Σ̂ + g.ŷ̄ * c.μ̂' + (g.S̄ + g.S̄') * c.H * c.Σ̂
end
_H_adjoint(::Val{true}, c, g) = _H_adjoint(c, g)
_H_adjoint(::Val{false}, c, g) = zero(c.H)

_c_adjoint(c, g) = g.ŷ̄
_c_adjoint(::Val{true}, c, g) = _c_adjoint(c, g)
_c_adjoint(::Val{false}, c, g) = zero(g.ŷ̄)

_R_adjoint(c, g) = g.S̄
_R_adjoint(::Val{true}, c, g) = _R_adjoint(c, g)
_R_adjoint(::Val{false}, c, g) = zero(g.S̄)

"""
    _kalman_adjoints(c, Δμ, ΔΣ, δll, dyn, obs)

Cotangents of the fused Kalman step from its cache `c` and output cotangents. Per-field
activity flags carried by the dynamics/observation components (see [`WithFlags`](@ref)) skip
the adjoints of fields that do not depend on the differentiated parameters; unwrapped atoms
are all-active.
"""
function _kalman_adjoints(c, Δμ, ΔΣ, δll, dyn, obs)
    fdyn = _field_flags(dyn)
    fobs = _field_flags(obs)
    g = _kalman_reverse_core(c, Δμ, ΔΣ, δll)
    # The primal symmetrises covariance expressions. Project their storage
    # cotangents too, including when the output seed selects only one triangle.
    return (;
        g.μ0̄,
        Σ0̄=symmetrise(g.Σ0̄),
        Ā=_A_adjoint(Val(fdyn[1]), c, g),
        b̄=_b_adjoint(Val(fdyn[2]), c, g),
        Q̄=symmetrise(_Q_adjoint(Val(fdyn[3]), c, g)),
        H̄=_H_adjoint(Val(fobs[1]), c, g),
        c̄=_c_adjoint(Val(fobs[2]), c, g),
        R̄=symmetrise(_R_adjoint(Val(fobs[3]), c, g)),
    )
end
