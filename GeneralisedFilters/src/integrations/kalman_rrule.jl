using ChainRulesCore: ChainRulesCore, NoTangent, @not_implemented

"""
    ChainRulesCore.rrule(::typeof(kf_loglikelihood), μ0, Σ0, As, bs, Qs, Hs, cs, Rs, ys)

Reverse-mode AD rule for the Kalman filter log-likelihood. The forward pass runs the KF
with gradient caching; the pullback computes analytical gradients using the backward recursion
from `kalman_gradient.jl`.
"""
function ChainRulesCore.rrule(
    ::typeof(kf_loglikelihood), μ0, Σ0, As, bs, Qs, Hs, cs, Rs, ys
)
    T = length(ys)

    # Forward pass with caching
    state = MvNormal(μ0, Σ0)
    caches = Vector{KalmanGradientCache}(undef, T)
    μ_prevs = Vector{typeof(μ0)}(undef, T)
    Σ_prevs = Vector{typeof(Σ0)}(undef, T)
    ll = zero(eltype(μ0))

    for t in 1:T
        μ_prevs[t], Σ_prevs[t] = params(state)
        state = kalman_predict(state, (As[t], bs[t], Qs[t]))
        state, ll_inc, caches[t] = _kalman_update_cached(
            state, Hs[t], cs[t], Rs[t], ys[t], nothing
        )
        ll += ll_inc
    end

    function kf_loglikelihood_pullback(Δll)
        # Backward pass: compute NLL gradients, then negate and scale by Δll
        ∂μ = zero(μ0)
        ∂Σ = zero(As[1])

        ∂As = similar(As)
        ∂bs = similar(bs)
        ∂Qs = similar(As)
        ∂Hs = similar(Hs)
        ∂cs = similar(cs)
        ∂Rs = Vector{typeof(zero(cs[1]) * zero(cs[1])')}(undef, T)

        for t in T:-1:1
            H, R = Hs[t], Rs[t]
            cache = caches[t]

            # Obs parameter gradients (NLL convention)
            ∂cs[t] = gradient_c(∂μ, cache)
            ∂Hs[t] = gradient_H(∂μ, ∂Σ, cache, cache.Σ_pred, Hs[t])
            ∂Rs[t] = gradient_R(∂μ, ∂Σ, cache)

            # Propagate through update step
            ∂μ_pred, ∂Σ_pred = backward_gradient_update(∂μ, ∂Σ, cache, H, R)

            # Dynamics parameter gradients (NLL convention)
            ∂bs[t] = gradient_b(∂μ_pred)
            ∂As[t] = gradient_A(∂μ_pred, ∂Σ_pred, μ_prevs[t], Σ_prevs[t], As[t])
            ∂Qs[t] = gradient_Q(∂Σ_pred)

            # Propagate through predict step
            ∂μ, ∂Σ = backward_gradient_predict(∂μ_pred, ∂Σ_pred, As[t])
        end

        # Initial state gradients
        ∂μ0_nll = ∂μ
        ∂Σ0_nll = ∂Σ

        # Convert NLL gradients → LL gradients and scale by Δll
        s = -Δll
        return (
            NoTangent(),
            s * ∂μ0_nll,
            s * ∂Σ0_nll,
            s .* ∂As,
            s .* ∂bs,
            s .* ∂Qs,
            s .* ∂Hs,
            s .* ∂cs,
            s .* ∂Rs,
            @not_implemented("Gradient w.r.t. observations not supported"),
        )
    end

    return ll, kf_loglikelihood_pullback
end
