using GaussianDistributions

export KalmanFilter, KF

struct KalmanFilter <: AbstractFilter end

KF() = KalmanFilter()

function initialise(
    rng::AbstractRNG, model::LinearGaussianStateSpaceModel, filter::KalmanFilter; kwargs...
)
    μ0, Σ0 = calc_initial(model.dyn; kwargs...)
    return Gaussian(μ0, Σ0)
end

function predict(
    rng::AbstractRNG,
    model::LinearGaussianStateSpaceModel,
    filter::KalmanFilter,
    step::Integer,
    states::Gaussian;
    kwargs...,
)
    μ, Σ = GaussianDistributions.pair(states)
    A, b, Q = calc_params(model.dyn, step; kwargs...)
    states = Gaussian(A * μ + b, A * Σ * A' + Q)
    return states
end

function update(
    model::LinearGaussianStateSpaceModel,
    filter::KalmanFilter,
    step::Integer,
    states::Gaussian,
    obs::AbstractVector;
    kwargs...,
)
    μ, Σ = GaussianDistributions.pair(states)
    H, c, R = calc_params(model.obs, step; kwargs...)

    # Update state
    m = H * μ + c
    y = obs - m
    S = H * Σ * H' + R
    K = Σ * H' / S

    states = Gaussian(μ + K * y, Σ - K * H * Σ)

    # Compute log-likelihood
    # HACK: force the covariance to be positive definite
    S = (S + S') / 2
    ll = logpdf(MvNormal(m, S), obs)

    return states, ll
end
