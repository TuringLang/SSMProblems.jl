export KalmanFilter, filter

struct KalmanFilter <: FilteringAlgorithm end

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
    y = obs - H * μ - c
    S = H * Σ * H' + R
    K = Σ * H' / S
    μ̂ = μ + K * y
    Σ̂ = Σ - K * H * Σ
    return (μ=μ̂, Σ=Σ̂)
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
    state = update(model, filter, step, state, obs, extra)
    return state
end

function filter(
    model::LinearGaussianStateSpaceModel{T},
    filter::KalmanFilter,
    data::Vector{Vector{T}},
    extras,
) where {T}
    μ0, Σ0 = calc_initial(model.dyn)
    state = (μ=μ0, Σ=Σ0)
    states = Vector{@NamedTuple{μ::Vector{T}, Σ::Matrix{T}}}(undef, length(data))
    for (i, obs) in enumerate(data[1:end])
        state = step(model, filter, i, state, obs, extras[i])
        states[i] = state
    end
    return states
end
