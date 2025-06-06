export KalmanFilter, filter, BatchKalmanFilter
using GaussianDistributions
using CUDA: i32
import LinearAlgebra: hermitianpart, transpose, Cholesky

export KalmanFilter, KF, KalmanSmoother, KS

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
    algo::KalmanFilter,
    iter::Integer,
    state::Gaussian,
    observation=nothing;
    kwargs...,
)
    μ, Σ = GaussianDistributions.pair(state)
    A, b, Q = calc_params(model.dyn, iter; kwargs...)
    return Gaussian(A * μ + b, A * Σ * transpose(A) + Q)
end

function update(
    model::LinearGaussianStateSpaceModel,
    algo::KalmanFilter,
    iter::Integer,
    state::Gaussian,
    observation::AbstractVector;
    kwargs...,
)
    μ, Σ = GaussianDistributions.pair(state)
    H, c, R = calc_params(model.obs, iter; kwargs...)

    # Update state
    m = H * μ + c
    y = observation - m
    S = H * Σ * transpose(H) + R
    S = (S + transpose(S)) / 2  # force symmetry
    S_chol = cholesky(S)
    KT = S_chol \ H * Σ  # TODO: only using `\` for better integration with CuSolver

    state = Gaussian(μ + transpose(KT) * y, Σ - transpose(KT) * H * Σ)

    # Compute log-likelihood
    ll = gaussian_likelihood(m, S, observation)

    return state, ll
end

function gaussian_likelihood(m::AbstractVector, S::AbstractMatrix, y::AbstractVector)
    return logpdf(MvNormal(m, S), y)
end

## KALMAN SMOOTHER #########################################################################

struct KalmanSmoother <: AbstractSmoother end

const KS = KalmanSmoother()

struct StateCallback{T} <: AbstractCallback
    proposed_states::Vector{Gaussian{Vector{T},Matrix{T}}}
    filtered_states::Vector{Gaussian{Vector{T},Matrix{T}}}
end
function StateCallback(N::Integer, T::Type)
    return StateCallback{T}(
        Vector{Gaussian{Vector{T},Matrix{T}}}(undef, N),
        Vector{Gaussian{Vector{T},Matrix{T}}}(undef, N),
    )
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
    model::LinearGaussianStateSpaceModel{T},
    algo::KalmanSmoother,
    observations::AbstractVector;
    t_smooth=1,
    callback=nothing,
    kwargs...,
) where {T}
    cache = StateCallback(length(observations), T)

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
    model::LinearGaussianStateSpaceModel{T},
    algo::KalmanSmoother,
    iter::Integer,
    back_state,
    obs;
    states_cache,
    kwargs...,
) where {T}
    μ, Σ = GaussianDistributions.pair(back_state)
    μ_pred, Σ_pred = GaussianDistributions.pair(states_cache.proposed_states[iter + 1])
    μ_filt, Σ_filt = GaussianDistributions.pair(states_cache.filtered_states[iter])

    G = Σ_filt * model.dyn.A' * inv(Σ_pred)
    μ = μ_filt .+ G * (μ .- μ_pred)
    Σ = Σ_filt .+ G * (Σ .- Σ_pred) * G'

    return Gaussian(μ, Σ)
end
