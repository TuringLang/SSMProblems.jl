export KalmanFilter, filter, BatchKalmanFilter
using CUDA: i32
import PDMats: PDMat

export KalmanFilter, KF, KalmanSmoother, KS

struct KalmanFilter <: AbstractFilter end

KF() = KalmanFilter()

function initialise(rng::AbstractRNG, prior::GaussianPrior, filter::KalmanFilter; kwargs...)
    μ0, Σ0 = calc_initial(prior; kwargs...)
    return GaussianDistribution(μ0, Σ0)
end

function predict(
    rng::AbstractRNG,
    dyn::LinearGaussianLatentDynamics,
    algo::KalmanFilter,
    iter::Integer,
    state::GaussianDistribution,
    observation=nothing;
    kwargs...,
)
    params = calc_params(dyn, iter; kwargs...)
    state = kalman_predict(state, params)
    return state
end

function kalman_predict(state, params)
    μ, Σ = mean_cov(state)
    A, b, Q = params

    μ̂ = A * μ + b
    Σ̂ = A * Σ * A' + Q
    return GaussianDistribution(μ̂, Σ̂)
end

function update(
    obs::LinearGaussianObservationProcess,
    algo::KalmanFilter,
    iter::Integer,
    state::GaussianDistribution,
    observation::AbstractVector;
    kwargs...,
)
    params = calc_params(obs, iter; kwargs...)
    state, ll = kalman_update(state, params, observation)
    return state, ll
end

function kalman_update(state, params, observation)
    μ, Σ = mean_cov(state)
    H, c, R = params

    # Update state
    m = H * μ + c
    y = observation - m
    S = H * Σ * H' + R
    S = (S + S') / 2  # force symmetric; TODO: replace with SA-compatibile hermitianpart
    S_chol = cholesky(S)
    K = Σ * H' / S_chol  # Zygote errors when using PDMat in solve

    state = GaussianDistribution(μ + K * y, Σ - K * H * Σ)

    # Compute log-likelihood
    ll = logpdf(MvNormal(m, PDMat(S_chol)), observation)

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

function backward(
    rng::AbstractRNG,
    model::LinearGaussianStateSpaceModel,
    algo::KalmanSmoother,
    iter::Integer,
    back_state,
    obs;
    states_cache,
    kwargs...,
)
    μ, Σ = mean_cov(back_state)
    μ_pred, Σ_pred = mean_cov(states_cache.proposed_states[iter + 1])
    μ_filt, Σ_filt = mean_cov(states_cache.filtered_states[iter])

    G = Σ_filt * model.dyn.A' * inv(Σ_pred)
    μ = μ_filt .+ G * (μ .- μ_pred)
    Σ = Σ_filt .+ G * (Σ .- Σ_pred) * G'

    return GaussianDistribution(μ, Σ)
end
