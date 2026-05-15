import PDMats: AbstractPDMat

export backward_gradient_update, backward_gradient_predict
export gradient_Q, gradient_R, gradient_A, gradient_b, gradient_H, gradient_c

# Each gradient helper computes the per-step contribution from THIS step's ll, scaled by
# `Δll` (the cotangent of the step's log-likelihood output), plus the propagation
# contribution from downstream cotangents of (μ_filt, Σ_filt). Parameters that enter
# only the state update (A, b, Q) have no `Δll`-scaled term; their gradient comes from
# the propagated cotangents alone, via `backward_gradient_update`.

## BACKWARD GRADIENT PROPAGATION ##############################################################

"""
    backward_gradient_update(∂μ_filt, ∂Σ_filt, Δll, z, S, I_KH, H, R)

Propagate cotangents backward through the Kalman update step (filtered → predicted).
The `Δll`-scaled local ∂ℓ/∂μ_pred and ∂ℓ/∂Σ_pred terms are added to the propagated
contributions from `∂μ_filt`, `∂Σ_filt`.
"""
function backward_gradient_update(
    ∂μ_filt::AbstractVector, ∂Σ_filt::AbstractMatrix, Δll, z, S, I_KH, H, R
)
    S_inv_z = S \ z
    ∂ℓ_∂μ_pred = H' * S_inv_z
    ∂ℓ_∂Σ_pred = -0.5 * (H' * (S \ H) - H' * (S_inv_z * S_inv_z') * H)

    ∂μ_pred = I_KH' * ∂μ_filt + Δll * ∂ℓ_∂μ_pred

    R_inv_z = R \ z
    cross_term = 0.5 * (∂μ_filt * (R_inv_z' * H) + (H' * R_inv_z) * ∂μ_filt')
    inner = ∂Σ_filt + cross_term
    ∂Σ_pred = I_KH' * inner * I_KH + Δll * ∂ℓ_∂Σ_pred

    return ∂μ_pred, ∂Σ_pred
end

"""
    backward_gradient_predict(∂μ_pred, ∂Σ_pred, A)

Propagate cotangents backward through the predict step (predicted → previous filtered).
"""
function backward_gradient_predict(∂μ_pred::AbstractVector, ∂Σ_pred::AbstractMatrix, A)
    return A' * ∂μ_pred, A' * ∂Σ_pred * A
end

## PARAMETER GRADIENTS ########################################################################

"""
    gradient_Q(∂Σ_pred)

`Q` enters only `Σ_pred` (additively); its cotangent is `∂Σ_pred`.
"""
gradient_Q(∂Σ_pred::AbstractMatrix) = ∂Σ_pred

"""
    gradient_b(∂μ_pred)

`b` enters only `μ_pred` (additively); its cotangent is `∂μ_pred`.
"""
gradient_b(∂μ_pred::AbstractVector) = ∂μ_pred

"""
    gradient_A(∂μ_pred, ∂Σ_pred, μ_prev, Σ_prev, A)

Chain rule through `μ_pred = A*μ_prev + b` and `Σ_pred = A*Σ_prev*A' + Q`.
"""
function gradient_A(
    ∂μ_pred::AbstractVector, ∂Σ_pred::AbstractMatrix, μ_prev::AbstractVector, Σ_prev, A
)
    return ∂μ_pred * μ_prev' + 2 * ∂Σ_pred * A * Σ_prev
end

"""
    gradient_c(∂μ_filt, Δll, z, S, K)

`c` enters `μ_filt = μ_pred + K*z` (via `z`) and the log-likelihood (via `z`).
"""
function gradient_c(∂μ_filt::AbstractVector, Δll, z, S, K)
    return Δll * (S \ z) - K' * ∂μ_filt
end

"""
    gradient_R(∂μ_filt, ∂Σ_filt, Δll, z, S, K)

`R` enters `Σ_filt` (via the Joseph form `K*R*K'`), the Kalman gain (cross term),
and the log-likelihood (via `S`).
"""
function gradient_R(∂μ_filt::AbstractVector, ∂Σ_filt::AbstractMatrix, Δll, z, S, K)
    S_inv_z = S \ z
    ∂ℓ_∂R = -0.5 * (inv(S) - S_inv_z * S_inv_z')
    cross_term = 0.5 * (K' * ∂μ_filt * S_inv_z' + S_inv_z * ∂μ_filt' * K)
    return K' * ∂Σ_filt * K - cross_term + Δll * ∂ℓ_∂R
end

"""
    gradient_H(∂μ_filt, ∂Σ_filt, Δll, μ_pred, μ_filt, z, S, K, I_KH, Σ_pred, H)

Chain rule through the innovation, Kalman gain, and information-form covariance update.
"""
function gradient_H(
    ∂μ_filt::AbstractVector,
    ∂Σ_filt::AbstractMatrix,
    Δll,
    μ_pred,
    μ_filt,
    z,
    S,
    K,
    I_KH,
    Σ_pred,
    H,
)
    S_inv_z = S \ z
    S_inv = inv(S)
    P_filt = I_KH * Σ_pred

    ∂ℓ_∂H = -(S_inv * H * Σ_pred - S_inv_z * μ_pred' - (S_inv_z * S_inv_z') * H * Σ_pred)
    ∂via_μ = S_inv_z * ∂μ_filt' * P_filt - K' * ∂μ_filt * μ_filt'
    ∂via_Σ = -2 * K' * ∂Σ_filt * P_filt

    return Δll * ∂ℓ_∂H + ∂via_μ + ∂via_Σ
end
