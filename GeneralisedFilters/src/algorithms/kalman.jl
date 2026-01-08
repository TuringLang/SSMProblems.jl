using CUDA: i32
import PDMats: PDMat, X_A_Xt, Xt_A_X, X_invA_Xt, Xt_invA_X
import LinearAlgebra: Symmetric

export KalmanFilter, KF, KalmanSmoother, KS
export BackwardInformationPredictor

"""
    KalmanFilter(; jitter=nothing)

Kalman filter for linear Gaussian state space models.

# Fields
- `jitter::Union{Nothing, Real}`: Optional value added to the covariance matrix after the
  update step to improve numerical stability. If `nothing`, no jitter is applied.
"""
struct KalmanFilter{T<:Union{Nothing,Real}} <: AbstractFilter
    jitter::T
end
KalmanFilter(; jitter=nothing) = KalmanFilter(jitter)

KF() = KalmanFilter()

function initialise(rng::AbstractRNG, prior::GaussianPrior, filter::KalmanFilter; kwargs...)
    μ0, Σ0 = calc_initial(prior; kwargs...)
    return MvNormal(μ0, Σ0)
end

function predict(
    rng::AbstractRNG,
    dyn::LinearGaussianLatentDynamics,
    algo::KalmanFilter,
    iter::Integer,
    state::MvNormal,
    observation=nothing;
    kwargs...,
)
    dyn_params = calc_params(dyn, iter; kwargs...)
    state = kalman_predict(state, dyn_params)
    return state
end

function kalman_predict(state, dyn_params)
    μ, Σ = params(state)
    A, b, Q = dyn_params

    μ̂ = A * μ + b
    Σ̂ = X_A_Xt(Σ, A) + Q
    return MvNormal(μ̂, Σ̂)
end

_compute_innovation(μ, H, c, y) = y - H * μ - c
_compute_innovation_cov(Σ, H, R) = PDMat(X_A_Xt(Σ, H) + R)
_compute_kalman_gain(Σ, H, S) = Σ * H' / S

function _compute_joseph_update(Σ, K, H, R)
    I_KH = I - K * H
    Σ̂_raw = X_A_Xt(Σ, I_KH) + X_A_Xt(R, K)
    return I_KH, Σ̂_raw
end

function _apply_jitter_and_wrap(Σ_raw, jitter)
    if !isnothing(jitter)
        Σ_raw = Σ_raw + jitter * I
    end
    return PDMat(Symmetric(Σ_raw))
end

function update(
    obs::LinearGaussianObservationProcess,
    algo::KalmanFilter,
    iter::Integer,
    state::MvNormal,
    observation::AbstractVector;
    kwargs...,
)
    obs_params = calc_params(obs, iter; kwargs...)
    state, ll = kalman_update(state, obs_params, observation, algo.jitter)
    return state, ll
end

function kalman_update(state, obs_params, observation, jitter)
    μ, Σ = params(state)
    H, c, R = obs_params

    z = _compute_innovation(μ, H, c, observation)
    S = _compute_innovation_cov(Σ, H, R)
    K = _compute_kalman_gain(Σ, H, S)
    _, Σ̂_raw = _compute_joseph_update(Σ, K, H, R)

    μ̂ = μ + K * z
    Σ̂ = _apply_jitter_and_wrap(Σ̂_raw, jitter)

    ll = logpdf(MvNormal(z, S), zero(z))

    return MvNormal(μ̂, Σ̂), ll
end

## KALMAN SMOOTHER #########################################################################

struct KalmanSmoother <: AbstractSmoother end

const KS = KalmanSmoother()

# Linear Gaussian implementation of backward_smooth.
# Uses the RTS equations: G = Σ_filt * A' * Σ_pred⁻¹
function backward_smooth(
    dyn::LinearGaussianLatentDynamics,
    algo::KalmanFilter,
    step::Integer,
    filtered::MvNormal,
    smoothed_next::MvNormal;
    predicted::Union{Nothing,MvNormal}=nothing,
    kwargs...,
)
    # Extract filtered and smoothed parameters
    μ_filt, Σ_filt = params(filtered)
    μ_smooth_next, Σ_smooth_next = params(smoothed_next)

    # Get dynamics parameters for the transition from step to step+1
    A, b, Q = calc_params(dyn, step + 1; kwargs...)

    # Compute predicted distribution if not provided
    if isnothing(predicted)
        μ_pred = A * μ_filt + b
        Σ_pred = X_A_Xt(Σ_filt, A) + Q
    else
        μ_pred, Σ_pred = params(predicted)
    end

    # Compute smoothing gain
    G = Σ_filt * A' / Σ_pred

    # RTS update
    μ_smooth = μ_filt + G * (μ_smooth_next - μ_pred)

    # Σ_pred - Σ_smooth_next may be singular (even though it is PSD) so we cannot use
    # X_A_Xt with Cholesky decomposition
    Σ_smooth = Σ_filt + G * (Σ_smooth_next - Σ_pred) * G'

    # Force symmetry and wrap in PDMat
    Σ_smooth = PDMat(Symmetric(Σ_smooth))

    return MvNormal(μ_smooth, Σ_smooth)
end

mutable struct StateCallback <: AbstractCallback
    proposed_states
    filtered_states
end

function (callback::StateCallback)(
    model::LinearGaussianStateSpaceModel,
    algo::KalmanFilter,
    state::T,
    observations,
    ::PostInitCallback;
    kwargs...,
) where {T}
    N = length(observations)
    callback.proposed_states = Vector{T}(undef, N)
    callback.filtered_states = Vector{T}(undef, N)
    return nothing
end

function (callback::StateCallback)(
    model::LinearGaussianStateSpaceModel,
    algo::KalmanFilter,
    iter::Integer,
    state,
    observation,
    ::PostPredictCallback;
    kwargs...,
)
    callback.proposed_states[iter] = deepcopy(state)
    return nothing
end

function (callback::StateCallback)(
    model::LinearGaussianStateSpaceModel,
    algo::KalmanFilter,
    iter::Integer,
    state,
    observation,
    ::PostUpdateCallback;
    kwargs...,
)
    callback.filtered_states[iter] = deepcopy(state)
    return nothing
end

"""
    smooth([rng,] model, algo, observations; t_smooth=1, callback=nothing, kwargs...)

Run a forward-backward smoothing pass to compute the smoothed distribution at time `t_smooth`.

This function first runs a forward filtering pass using the Kalman filter, caching all
filtered distributions, then performs backward smoothing using the Rauch-Tung-Striebel
equations from time T back to `t_smooth`.

# Arguments
- `rng`: Random number generator (optional, defaults to `default_rng()`)
- `model`: A linear Gaussian state space model
- `algo`: The smoothing algorithm (e.g., `KalmanSmoother()`)
- `observations`: Vector of observations y₁, ..., yₜ

# Keyword Arguments
- `t_smooth=1`: The time step at which to return the smoothed distribution
- `callback=nothing`: Optional callback for the forward filtering pass

# Returns
A tuple `(smoothed, log_likelihood)` where:
- `smoothed`: The smoothed distribution p(xₜ | y₁:ₜ) at time `t_smooth`
- `log_likelihood`: The total log-likelihood from the forward pass

# Example
```julia
model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
smoothed, ll = smooth(model, KalmanSmoother(), observations)
```

See also: [`backward_smooth`](@ref), [`filter`](@ref)
"""
function smooth(
    rng::AbstractRNG,
    model::LinearGaussianStateSpaceModel,
    algo::KalmanSmoother,
    observations::AbstractVector;
    t_smooth=1,
    callback=nothing,
    kwargs...,
)
    cache = StateCallback(nothing, nothing)
    filtered, ll = filter(
        rng, model, KalmanFilter(), observations; callback=cache, kwargs...
    )

    smoothed = filtered
    for t in (length(observations) - 1):-1:t_smooth
        smoothed = backward_smooth(
            dyn(model),
            KalmanFilter(),
            t,
            cache.filtered_states[t],
            smoothed;
            predicted=cache.proposed_states[t + 1],
            kwargs...,
        )
    end

    return smoothed, ll
end

## BACKWARD INFORMATION FILTER #############################################################

"""
    BackwardInformationPredictor(; jitter=nothing, initial_jitter=nothing)

An algorithm to recursively compute the predictive likelihood p(y_{t:T} | x_t) of a linear
Gaussian state space model in information form.

# Fields
- `jitter::Union{Nothing, Real}`: Optional value added to the precision matrix Ω after the
  backward predict step to improve numerical stability. If `nothing`, no jitter is applied.
- `initial_jitter::Union{Nothing, Real}`: Optional value added to the precision matrix Ω
  at initialization to improve numerical stability.

This implementation is based on https://arxiv.org/pdf/1505.06357
"""
struct BackwardInformationPredictor{T0,T} <: AbstractBackwardPredictor
    initial_jitter::T0
    jitter::T
end
function BackwardInformationPredictor(; initial_jitter=nothing, jitter=nothing)
    return BackwardInformationPredictor(initial_jitter, jitter)
end

"""
    backward_initialise(rng, obs, algo, iter, y; kwargs...)

Initialise a backward predictor at time `T` with observation `y`, forming the likelihood
p(y_T | x_T).
"""
function backward_initialise(
    rng::AbstractRNG,
    obs::LinearGaussianObservationProcess,
    filter::BackwardInformationPredictor,
    iter::Integer,
    y;
    kwargs...,
)
    H, c, R = calc_params(obs, iter; kwargs...)
    R_inv = inv(R)
    λ = H' * R_inv * (y - c)
    Ω = Xt_A_X(R_inv, H)

    # Optionally add initial_jitter for numerical stability and convert to PDMat
    if !isnothing(filter.initial_jitter)
        Ω += filter.initial_jitter * I
    end
    Ω = PDMat(Symmetric(Ω))

    return InformationLikelihood(λ, Ω)
end

"""
    backward_predict(rng, dyn, algo, iter, state; prev_outer=nothing, next_outer=nothing, kwargs...)

Perform a backward prediction step to compute p(y_{t+1:T} | x_t) from p(y_{t:T} | x_{t+1}).
"""
function backward_predict(
    rng::AbstractRNG,
    dyn::LinearGaussianLatentDynamics,
    algo::BackwardInformationPredictor,
    iter::Integer,
    state::InformationLikelihood;
    kwargs...,
)
    λ, Ω = natural_params(state)
    # Use iter + 1 to get transition from x_iter to x_{iter+1}, matching forward filter convention
    A, b, Q = calc_params(dyn, iter + 1; kwargs...)
    F = cholesky(Q).L

    m = λ - Ω * b
    # HACK: missing method for Symmetir{SMatrix} + UniformScaling
    Λ = PDMat(Symmetric(Xt_A_X(Ω, F).data + I))

    # Ω̂ = A' * (I - Ω * F * inv(Λ) * F') * Ω * A
    # λ̂ = A' * (I - Ω * F * inv(Λ) * F') * m
    FΛ_inv_Ft = X_invA_Xt(Λ, F)
    I_minus_term = I - Ω * FΛ_inv_Ft
    Ω̂ = A' * I_minus_term * Ω * A
    λ̂ = A' * I_minus_term * m

    # Optionally add jitter for numerical stability and convert to PDMat
    if !isnothing(algo.jitter)
        Ω̂ += algo.jitter * I
    end
    Ω̂ = PDMat(Symmetric(Ω̂))

    return InformationLikelihood(λ̂, Ω̂)
end

"""
    backward_update(obs, algo, iter, state, y; kwargs...)

Incorporate an observation `y` at time `t` to compute p(y_{t:T} | x_t) from p(y_{t+1:T} | x_t).
"""
function backward_update(
    obs::LinearGaussianObservationProcess,
    algo::BackwardInformationPredictor,
    iter::Integer,
    state::InformationLikelihood,
    y;
    kwargs...,
)
    λ, Ω = natural_params(state)
    H, c, R = calc_params(obs, iter; kwargs...)
    R_inv = inv(R)

    λ̂ = λ + H' * R_inv * (y - c)
    Ω̂ = PDMat(Ω + Xt_A_X(R_inv, H))

    return InformationLikelihood(λ̂, Ω̂)
end

## TWO-FILTER SMOOTHING ####################################################################

# Linear Gaussian implementation of two_filter_smooth.
# Combines distributions using the product of Gaussians in information form:
#   J_smooth = Σ⁻¹ + Ω,  h_smooth = Σ⁻¹μ + λ
function two_filter_smooth(filtered::MvNormal, backward_lik::InformationLikelihood)
    μ_filt, Σ_filt = params(filtered)
    λ_back, Ω_back = natural_params(backward_lik)

    # Convert filtered distribution to information form
    Σ_filt_inv = inv(Σ_filt)
    λ_filt = Σ_filt_inv * μ_filt

    # Combine in information form (product of Gaussians)
    Ω_smooth = PDMat(Σ_filt_inv + Ω_back)
    λ_smooth = λ_filt + λ_back

    # Convert back to moment form
    Σ_smooth = inv(Ω_smooth)
    μ_smooth = Σ_smooth * λ_smooth

    return MvNormal(μ_smooth, Σ_smooth)
end
