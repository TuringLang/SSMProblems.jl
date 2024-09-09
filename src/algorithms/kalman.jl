export KalmanFilter, filter

struct KalmanFilter <: FilteringAlgorithm end

function initialise(
    model::LinearGaussianStateSpaceModel{T}, filter::KalmanFilter, extra
) where {T}
    μ0, Σ0 = calc_initial(model.dyn, extra)
    return (μ=μ0, Σ=Σ0)
end

function predict(
    model::LinearGaussianStateSpaceModel{T},
    filter::KalmanFilter,
    step::Integer,
    state::@NamedTuple{μ::Vector{T}, Σ::Matrix{T}},
    extra,
) where {T}
    μ, Σ = state.μ, state.Σ
    A, b, Q = calc_params(model.dyn, step, extra)
    μ̂ = A * μ + b
    Σ̂ = A * Σ * A' + Q
    return (μ=μ̂, Σ=Σ̂)
end

function update(
    model::LinearGaussianStateSpaceModel{T},
    filter::KalmanFilter,
    step::Integer,
    state::@NamedTuple{μ::Vector{T}, Σ::Matrix{T}},
    obs::Vector{T},
    extra,
) where {T}
    μ, Σ = state.μ, state.Σ
    H, c, R = calc_params(model.obs, step, extra)

    # Update state
    m = H * μ + c
    y = obs - m
    S = H * Σ * H' + R
    K = Σ * H' / S
    μ̂ = μ + K * y
    Σ̂ = Σ - K * H * Σ

    # Compute log-likelihood
    ll = logpdf(MvNormal(m, S), obs)

    return (μ=μ̂, Σ=Σ̂), ll
end

function step(
    model::LinearGaussianStateSpaceModel{T},
    filter::KalmanFilter,
    step::Integer,
    state::@NamedTuple{μ::Vector{T}, Σ::Matrix{T}},
    obs::Vector{T},
    extra,
) where {T}
    state = predict(model, filter, step, state, extra)
    state, ll = update(model, filter, step, state, obs, extra)
    return state, ll
end

function filter(
    model::LinearGaussianStateSpaceModel{T},
    filter::KalmanFilter,
    data::Vector{Vector{T}},
    extra0,
    extras,
) where {T}
    state = initialise(model, filter, extra0)
    states = Vector{@NamedTuple{μ::Vector{T}, Σ::Matrix{T}}}(undef, length(data))
    ll = 0.0
    for (i, obs) in enumerate(data)
        state, step_ll = step(model, filter, i, state, obs, extras[i])
        states[i] = state
        ll += step_ll
    end
    return states, ll
end
