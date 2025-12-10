export KalmanFilter, filter
using CUDA: i32
import PDMats: PDMat, X_A_Xt, Xt_A_X, X_invA_Xt, Xt_invA_X
import LinearAlgebra: Symmetric

export KalmanFilter, KF, KalmanSmoother, KS
export BackwardInformationPredictor
export backward_initialise, backward_predict, backward_update

"""
    KalmanFilter(; jitter=nothing)

Kalman filter for linear Gaussian state space models.

# Fields
- `jitter::Union{Nothing, Real}`: Optional value added to the covariance matrix after the
  update step to improve numerical stability. If `nothing`, no jitter is applied.
"""
struct KalmanFilter <: AbstractFilter
    jitter::Union{Nothing,Real}
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

    # Compute innovation distribution
    m = H * μ + c
    S = PDMat(X_A_Xt(Σ, H) + R)
    ȳ = observation - m
    K = Σ * H' / S

    # Update parameters using Joseph form to ensure numerical stability
    μ̂ = μ + K * ȳ
    Σ̂ = X_A_Xt(Σ, I - K * H) + X_A_Xt(R, K)

    # Optionally add jitter for numerical stability and convert to PDMat
    if !isnothing(jitter)
        for i in 1:size(Σ̂, 1)
            Σ̂[i, i] += jitter
        end
    end
    Σ̂ = PDMat(Symmetric(Σ̂))

    state = MvNormal(μ̂, Σ̂)

    # Compute log-likelihood
    ll = logpdf(MvNormal(m, S), observation)

    return state, ll
end

## KALMAN SMOOTHER #########################################################################

struct KalmanSmoother <: AbstractSmoother end

const KS = KalmanSmoother()

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

    back_state = filtered
    for t in (length(observations) - 1):-1:t_smooth
        back_state = backward(
            rng, model, algo, t, back_state, observations[t]; states_cache=cache, kwargs...
        )
    end

    return back_state, ll
end

import LinearAlgebra: eigen

function backward(
    rng::AbstractRNG,
    model::LinearGaussianStateSpaceModel,
    algo::KalmanSmoother,
    iter::Integer,
    back_state::MvNormal,
    obs;
    states_cache,
    kwargs...,
)
    # Extract filtered and predicted states
    μ_filt, Σ_filt = params(states_cache.filtered_states[iter])
    μ_pred, Σ_pred = params(states_cache.proposed_states[iter + 1])
    μ_back, Σ_back = params(back_state)

    dyn_params = calc_params(model.dyn, iter + 1; kwargs...)
    A, b, Q = dyn_params

    G = Σ_filt * A' / Σ_pred
    μ̂ = μ_filt + G * (μ_back - μ_pred)

    # Σ_pred - Σ_back  may be singular (even though it is PSD) so cannot use X_A_Xt with Cholesky
    Σ̂ = Σ_filt + G * (Σ_back - Σ_pred) * G'

    # Force symmetry
    Σ̂ = PDMat(Symmetric(Σ̂))

    return MvNormal(μ̂, Σ̂)
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
struct BackwardInformationPredictor <: AbstractBackwardPredictor
    jitter::Union{Nothing,Real}
    initial_jitter::Union{Nothing,Real}
end
function BackwardInformationPredictor(; jitter=nothing, initial_jitter=nothing)
    return BackwardInformationPredictor(jitter, initial_jitter)
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
        for i in 1:size(Ω, 1)
            Ω[i, i] += filter.initial_jitter
        end
    end
    println(eigen(Ω).values)
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
    A, b, Q = calc_params(dyn, iter; kwargs...)
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
        for i in 1:size(Ω̂, 1)
            Ω̂[i, i] += algo.jitter
        end
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
