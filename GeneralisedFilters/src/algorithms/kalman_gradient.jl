import PDMats: AbstractPDMat

export KalmanGradientCache
export update_with_cache
export backward_gradient_update, backward_gradient_predict
export gradient_Q, gradient_R, gradient_A, gradient_b, gradient_H, gradient_c, gradient_y

"""
    KalmanGradientCache

Cache of intermediate values from a Kalman filter update step for gradient computation.

# Fields
- `μ_pred`: Predicted mean x̂_{n|n-1}
- `Σ_pred`: Predicted covariance P_{n|n-1}
- `μ_filt`: Filtered mean x̂_{n|n}
- `Σ_filt`: Filtered covariance P_{n|n}
- `S`: Innovation covariance
- `K`: Kalman gain
- `z`: Innovation (y - H*μ_pred - c)
- `I_KH`: I - K*H
"""
struct KalmanGradientCache{μpT,ΣpT,μfT,ΣfT,ST,KT,zT,IKT}
    μ_pred::μpT
    Σ_pred::ΣpT
    μ_filt::μfT
    Σ_filt::ΣfT
    S::ST
    K::KT
    z::zT
    I_KH::IKT
end

"""
    update_with_cache(obs, algo, iter, state, observation; kwargs...)

Perform Kalman update and return cache for gradient computation.

This extends the standard update step to also return a `KalmanGradientCache` containing
intermediate values needed for efficient backward gradient propagation.

# Returns
A tuple `(filtered_state, log_likelihood, cache)` where:
- `filtered_state`: The posterior state as `MvNormal`
- `log_likelihood`: The log-likelihood increment
- `cache`: A `KalmanGradientCache` for use in backward gradient computation
"""
function update_with_cache(
    obs::LinearGaussianObservationProcess,
    algo::KalmanFilter,
    iter::Integer,
    state::MvNormal,
    observation::AbstractVector;
    kwargs...,
)
    μ_pred, Σ_pred = params(state)
    H, c, R = calc_params(obs, iter; kwargs...)

    z = _compute_innovation(μ_pred, H, c, observation)
    S = _compute_innovation_cov(Σ_pred, H, R)
    K = _compute_kalman_gain(Σ_pred, H, S)
    I_KH, Σ_filt_raw = _compute_joseph_update(Σ_pred, K, H, R)

    μ_filt = μ_pred + K * z
    Σ_filt = _apply_jitter_and_wrap(Σ_filt_raw, algo.jitter)

    filtered_state = MvNormal(μ_filt, Σ_filt)
    ll = logpdf(MvNormal(H * μ_pred + c, S), observation)

    cache = KalmanGradientCache(μ_pred, Σ_pred, μ_filt, Σ_filt, S, K, z, I_KH)

    return filtered_state, ll, cache
end

## BACKWARD GRADIENT PROPAGATION ##############################################################

"""
    backward_gradient_update(∂μ_filt, ∂Σ_filt, cache, H, R)

Propagate gradients backward through the Kalman update step (filtered → predicted).

This implements equations 8-9 from Parellier et al., computing the gradients with respect
to the predicted state from the gradients with respect to the filtered state.

# Arguments
- `∂μ_filt`: Gradient of loss w.r.t. filtered mean ∂L/∂x̂_{n|n}
- `∂Σ_filt`: Gradient of loss w.r.t. filtered covariance ∂L/∂P_{n|n}
- `cache`: `KalmanGradientCache` from the forward pass
- `H`: Observation matrix at this time step
- `R`: Observation noise covariance at this time step

# Returns
A tuple `(∂μ_pred, ∂Σ_pred)` containing gradients w.r.t. the predicted state.
"""
function backward_gradient_update(
    ∂μ_filt::AbstractVector, ∂Σ_filt::AbstractMatrix, cache::KalmanGradientCache, H, R
)
    z, S, I_KH = cache.z, cache.S, cache.I_KH

    # NLL local derivatives (standard convention with 1/2 factor)
    S_inv_z = S \ z
    ∂l_∂μ_pred = -H' * S_inv_z
    ∂l_∂Σ_pred = 0.5 * (H' * (S \ H) - H' * (S_inv_z * S_inv_z') * H)

    # Equation 8: ∂L/∂μ_pred = (I-KH)' * ∂L/∂μ_filt + ∂l/∂μ_pred
    ∂μ_pred = I_KH' * ∂μ_filt + ∂l_∂μ_pred

    # Equation 9: ∂L/∂Σ_pred = (I-KH)' * [∂L/∂Σ_filt + cross_term] * (I-KH) + ∂l/∂Σ_pred
    R_inv_z = R \ z
    cross_term = 0.5 * (∂μ_filt * (R_inv_z' * H) + (H' * R_inv_z) * ∂μ_filt')
    inner = ∂Σ_filt + cross_term
    ∂Σ_pred = I_KH' * inner * I_KH + ∂l_∂Σ_pred

    return ∂μ_pred, ∂Σ_pred
end

"""
    backward_gradient_predict(∂μ_pred, ∂Σ_pred, A)

Propagate gradients backward through the Kalman predict step (predicted → previous filtered).

This implements equations 10-11 from Parellier et al.

# Arguments
- `∂μ_pred`: Gradient of loss w.r.t. predicted mean ∂L/∂x̂_{n|n-1}
- `∂Σ_pred`: Gradient of loss w.r.t. predicted covariance ∂L/∂P_{n|n-1}
- `A`: Dynamics matrix at this time step

# Returns
A tuple `(∂μ_filt_prev, ∂Σ_filt_prev)` containing gradients w.r.t. the previous filtered state.
"""
function backward_gradient_predict(∂μ_pred::AbstractVector, ∂Σ_pred::AbstractMatrix, A)
    ∂μ_filt_prev = A' * ∂μ_pred       # Equation 10
    ∂Σ_filt_prev = A' * ∂Σ_pred * A   # Equation 11
    return ∂μ_filt_prev, ∂Σ_filt_prev
end

## PARAMETER GRADIENTS ########################################################################

"""
    gradient_Q(∂Σ_pred)

Compute gradient of NLL w.r.t. process noise covariance Q.

Implements equation 13 from Parellier et al.: ∂L/∂Q = ∂L/∂P_{n|n-1}
"""
function gradient_Q(∂Σ_pred::AbstractMatrix)
    return ∂Σ_pred
end

"""
    gradient_R(∂μ_filt, ∂Σ_filt, cache)

Compute gradient of NLL w.r.t. observation noise covariance R.

Implements equation 14 from Parellier et al.
"""
function gradient_R(
    ∂μ_filt::AbstractVector, ∂Σ_filt::AbstractMatrix, cache::KalmanGradientCache
)
    z, S, K = cache.z, cache.S, cache.K
    S_inv_z = S \ z

    # Local NLL derivative: ∂l/∂R = 0.5 * (S⁻¹ - S⁻¹zz'S⁻¹)
    ∂l_∂R = 0.5 * (inv(S) - S_inv_z * S_inv_z')

    # Equation 14: ∂L/∂R = K'*∂L/∂Σ_filt*K - cross_term + ∂l/∂R
    cross_term = 0.5 * (K' * ∂μ_filt * S_inv_z' + S_inv_z * ∂μ_filt' * K)
    return K' * ∂Σ_filt * K - cross_term + ∂l_∂R
end

"""
    gradient_y(∂μ_filt, cache)

Compute gradient of NLL w.r.t. observation y.

Implements equation 12 from Parellier et al.: ∂L/∂y = K'*∂L/∂μ_filt + ∂l/∂y
"""
function gradient_y(∂μ_filt::AbstractVector, cache::KalmanGradientCache)
    z, S, K = cache.z, cache.S, cache.K
    ∂l_∂y = S \ z
    return K' * ∂μ_filt + ∂l_∂y
end

"""
    gradient_A(∂μ_pred, ∂Σ_pred, μ_prev, Σ_prev, A)

Compute gradient of NLL w.r.t. dynamics matrix A.

Derived via chain rule through μ_pred = A*μ_prev + b and Σ_pred = A*Σ_prev*A' + Q.
"""
function gradient_A(
    ∂μ_pred::AbstractVector, ∂Σ_pred::AbstractMatrix, μ_prev::AbstractVector, Σ_prev, A
)
    # ∂L/∂A = ∂L/∂μ_pred * μ_prev' + 2 * ∂L/∂Σ_pred * A * Σ_prev
    return ∂μ_pred * μ_prev' + 2 * ∂Σ_pred * A * Σ_prev
end

"""
    gradient_b(∂μ_pred)

Compute gradient of NLL w.r.t. dynamics offset b.

Derived via chain rule through μ_pred = A*μ_prev + b.
"""
function gradient_b(∂μ_pred::AbstractVector)
    return ∂μ_pred
end

"""
    gradient_H(∂μ_filt, ∂Σ_filt, cache, Σ_pred)

Compute gradient of NLL w.r.t. observation matrix H.

Derived via chain rule through z = y - H*μ_pred - c, S = H*Σ_pred*H' + R, and the update.
"""
function gradient_H(
    ∂μ_filt::AbstractVector, ∂Σ_filt::AbstractMatrix, cache::KalmanGradientCache, Σ_pred
)
    μ_pred, z, S, K, I_KH = cache.μ_pred, cache.z, cache.S, cache.K, cache.I_KH
    S_inv_z = S \ z
    S_inv = S \ I

    # Local NLL derivative: l = 0.5*(log|S| + z'S⁻¹z)
    # ∂l/∂H = S⁻¹*H*Σ_pred - S⁻¹*z*μ_pred' - S⁻¹*z*z'*S⁻¹*H*Σ_pred
    ∂l_∂H = S_inv * Σ_pred - S_inv_z * μ_pred' - (S_inv_z * S_inv_z') * Σ_pred

    # Contribution through filtered mean: μ_filt = μ_pred + K*z
    # ∂z/∂H = -μ_pred' (outer product form for each element)
    ∂via_μ = -K' * ∂μ_filt * μ_pred'

    # Contribution through filtered covariance via I_KH
    ∂via_Σ = -K' * ∂Σ_filt * I_KH * μ_pred' - I_KH' * ∂Σ_filt * K * μ_pred'

    return ∂l_∂H + ∂via_μ + ∂via_Σ
end

"""
    gradient_c(∂μ_filt, cache)

Compute gradient of NLL w.r.t. observation offset c.

Derived via chain rule through z = y - H*μ_pred - c.
"""
function gradient_c(∂μ_filt::AbstractVector, cache::KalmanGradientCache)
    z, S, K = cache.z, cache.S, cache.K
    ∂l_∂c = -(S \ z)  # ∂l/∂z * ∂z/∂c = -∂l/∂z
    return ∂l_∂c - K' * ∂μ_filt
end
