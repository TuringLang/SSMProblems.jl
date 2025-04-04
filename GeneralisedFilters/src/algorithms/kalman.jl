export KalmanFilter, filter, BatchKalmanFilter
using GaussianDistributions
using CUDA: i32
import LinearAlgebra: hermitianpart

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
    return Gaussian(A * μ + b, A * Σ * A' + Q)
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
    S = hermitianpart(H * Σ * H' + R)
    K = Σ * H' / S

    state = Gaussian(μ + K * y, Σ - K * H * Σ)

    # Compute log-likelihood
    ll = logpdf(MvNormal(m, S), observation)

    return state, ll
end

struct BatchKalmanFilter <: AbstractBatchFilter
    batch_size::Int
end

function initialise(
    rng::AbstractRNG,
    model::LinearGaussianStateSpaceModel{T},
    algo::BatchKalmanFilter;
    kwargs...,
) where {T}
    μ0s, Σ0s = batch_calc_initial(model.dyn, algo.batch_size; kwargs...)
    return BatchGaussianDistribution(μ0s, Σ0s)
end

function predict(
    rng::AbstractRNG,
    model::LinearGaussianStateSpaceModel{T},
    algo::BatchKalmanFilter,
    iter::Integer,
    state::BatchGaussianDistribution,
    observation;
    kwargs...,
) where {T}
    μs, Σs = state.μs, state.Σs
    As, bs, Qs = batch_calc_params(model.dyn, iter, algo.batch_size; kwargs...)
    μ̂s = NNlib.batched_vec(As, μs) .+ bs
    Σ̂s = NNlib.batched_mul(NNlib.batched_mul(As, Σs), NNlib.batched_transpose(As)) .+ Qs
    return BatchGaussianDistribution(μ̂s, Σ̂s)
end

function update(
    model::LinearGaussianStateSpaceModel{T},
    algo::BatchKalmanFilter,
    iter::Integer,
    state::BatchGaussianDistribution,
    observation;
    kwargs...,
) where {T}
    μs, Σs = state.μs, state.Σs
    Hs, cs, Rs = batch_calc_params(model.obs, iter, algo.batch_size; kwargs...)
    D = size(observation, 1)

    m = NNlib.batched_vec(Hs, μs) .+ cs
    y_res = cu(observation) .- m
    S = NNlib.batched_mul(Hs, NNlib.batched_mul(Σs, NNlib.batched_transpose(Hs))) .+ Rs

    ΣH_T = NNlib.batched_mul(Σs, NNlib.batched_transpose(Hs))

    S_inv = CUDA.similar(S)
    d_ipiv, _, d_S = CUDA.CUBLAS.getrf_strided_batched(S, true)
    CUDA.CUBLAS.getri_strided_batched!(d_S, S_inv, d_ipiv)

    diags = CuArray{eltype(S)}(undef, size(S, 1), size(S, 3))
    for i in 1:size(S, 1)
        diags[i, :] .= d_S[i, i, :]
    end

    log_dets = sum(log ∘ abs, diags; dims=1)

    K = NNlib.batched_mul(ΣH_T, S_inv)

    μ_filt = μs .+ NNlib.batched_vec(K, y_res)
    Σ_filt = Σs .- NNlib.batched_mul(K, NNlib.batched_mul(Hs, Σs))

    inv_term = NNlib.batched_vec(S_inv, y_res)
    log_likes = -T(0.5) * NNlib.batched_vec(reshape(y_res, 1, D, size(S, 3)), inv_term)
    log_likes = log_likes .- T(0.5) * (log_dets .+ D * log(T(2π)))

    # HACK: only errors seems to be from numerical stability so will just overwrite
    log_likes[isnan.(log_likes)] .= -Inf

    return BatchGaussianDistribution(μ_filt, Σ_filt), dropdims(log_likes; dims=1)
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
