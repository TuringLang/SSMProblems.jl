export DiscreteFilter, DF
export BackwardDiscretePredictor
export DiscreteSmoother

"""
    DiscreteFilter <: AbstractFilter

Forward filtering algorithm for discrete (finite) state space models.

Computes the filtered distribution π_t(i) = p(x_t = i | y_{1:t}) recursively.
"""
struct DiscreteFilter <: AbstractFilter end
const DF = DiscreteFilter

function initialise(rng::AbstractRNG, prior::DiscretePrior, ::DiscreteFilter; kwargs...)
    return calc_α0(prior; kwargs...)
end

function predict(
    rng::AbstractRNG,
    dyn::DiscreteLatentDynamics,
    filter::DiscreteFilter,
    step::Integer,
    states::AbstractVector,
    observation;
    kwargs...,
)
    P = calc_P(dyn, step; kwargs...)
    return (states' * P)'
end

function update(
    obs::ObservationProcess,
    filter::DiscreteFilter,
    step::Integer,
    states::AbstractVector,
    observation;
    kwargs...,
)
    # Compute emission probability vector
    # TODO: should we define density as part of the interface or run the whole algorithm in
    # log space?
    b = map(
        x -> exp(SSMProblems.logdensity(obs, step, x, observation; kwargs...)),
        eachindex(states),
    )
    filtered_states = b .* states
    likelihood = sum(filtered_states)
    return (filtered_states / likelihood), log(likelihood)
end

## BACKWARD DISCRETE PREDICTOR #############################################################

"""
    BackwardDiscretePredictor <: AbstractBackwardPredictor

Algorithm to recursively compute the backward likelihood β_t(i) = p(y_{t:T} | x_t = i)
for discrete state space models.

All computations are performed in log-space using logsumexp for numerical stability.
The resulting `DiscreteLikelihood` stores log β values internally.
"""
struct BackwardDiscretePredictor <: AbstractBackwardPredictor end

"""
    backward_initialise(rng, obs, algo::BackwardDiscretePredictor, iter, y; kwargs...)

Initialize the backward likelihood at time T with observation y_T.

Returns a `DiscreteLikelihood` where log β_T(i) = log p(y_T | x_T = i).
"""
function backward_initialise(
    rng::AbstractRNG,
    obs::ObservationProcess,
    ::BackwardDiscretePredictor,
    iter::Integer,
    y;
    num_states::Integer,
    kwargs...,
)
    log_β = map(i -> SSMProblems.logdensity(obs, iter, i, y; kwargs...), 1:num_states)
    return DiscreteLikelihood(log_β)
end

"""
    backward_predict(rng, dyn, algo::BackwardDiscretePredictor, iter, state; kwargs...)

Backward prediction step: marginalize through dynamics without incorporating observations.

Takes p(y_{t+1:T} | x_{t+1}) and computes p(y_{t+1:T} | x_t) by marginalizing over x_{t+1}:
    p(y_{t+1:T} | x_t = i) = Σ_j P_{ij} p(y_{t+1:T} | x_{t+1} = j)

In log-space: log p(y_{t+1:T} | x_t = i) = logsumexp_j(log P_{ij} + log p(y_{t+1:T} | x_{t+1} = j))
"""
function backward_predict(
    rng::AbstractRNG,
    dyn::DiscreteLatentDynamics,
    ::BackwardDiscretePredictor,
    iter::Integer,
    state::DiscreteLikelihood;
    kwargs...,
)
    log_β_next = log_likelihoods(state)
    P = calc_P(dyn, iter + 1; kwargs...)
    K = length(log_β_next)

    log_β = map(1:K) do i
        logsumexp(log.(P[i, :]) .+ log_β_next)
    end

    return DiscreteLikelihood(log_β)
end

"""
    backward_update(obs, algo::BackwardDiscretePredictor, iter, state, y; kwargs...)

Incorporate observation y_t into the backward likelihood.

Updates: log β(i) += log p(y_t | x_t = i)

This transforms p(y_{t+1:T} | x_t) into β_t = p(y_{t:T} | x_t).
"""
function backward_update(
    obs::ObservationProcess,
    ::BackwardDiscretePredictor,
    iter::Integer,
    state::DiscreteLikelihood,
    y;
    kwargs...,
)
    log_β = log_likelihoods(state)
    K = length(log_β)

    log_emission = map(i -> SSMProblems.logdensity(obs, iter, i, y; kwargs...), 1:K)
    log_β_new = log_β .+ log_emission

    return DiscreteLikelihood(log_β_new)
end

## DISCRETE SMOOTHER #######################################################################

"""
    DiscreteSmoother <: AbstractSmoother

A forward-backward smoother for discrete state space models.
"""
struct DiscreteSmoother <: AbstractSmoother end

"""
    backward_smooth(dyn, algo::DiscreteFilter, step, filtered, smoothed_next; predicted, kwargs...)

Perform one step of backward smoothing for discrete state spaces.

Computes γ_t(i) = π_t(i) * Σ_j [P_{ij} * γ_{t+1}(j) / π̂_{t+1}(j)]

where:
- π_t(i) is the filtered distribution at time t
- γ_{t+1}(j) is the smoothed distribution at time t+1
- π̂_{t+1}(j) is the predicted distribution at time t+1
- P_{ij} is the transition probability from state i to state j
"""
function backward_smooth(
    dyn::DiscreteLatentDynamics,
    ::DiscreteFilter,
    step::Integer,
    filtered::AbstractVector,
    smoothed_next::AbstractVector;
    predicted::AbstractVector,
    kwargs...,
)
    P = calc_P(dyn, step + 1; kwargs...)
    K = length(filtered)

    smoothed = map(1:K) do i
        correction = sum(1:K) do j
            P[i, j] * smoothed_next[j] / predicted[j]
        end
        filtered[i] * correction
    end

    return smoothed
end

"""
    smooth(rng, model::DiscreteStateSpaceModel, algo::DiscreteSmoother, observations; t_smooth=1, kwargs...)

Run forward-backward smoothing for discrete state space models.

Returns the smoothed distribution at time `t_smooth` and the log-likelihood.
"""
function smooth(
    rng::AbstractRNG,
    model::DiscreteStateSpaceModel,
    ::DiscreteSmoother,
    observations::AbstractVector;
    t_smooth=1,
    kwargs...,
)
    T = length(observations)
    df = DiscreteFilter()

    # Forward pass: store filtered and predicted distributions
    filtered = Vector{Vector{Float64}}(undef, T)
    predicted = Vector{Vector{Float64}}(undef, T)

    total_ll = 0.0
    state = let s = initialise(rng, prior(model), df; kwargs...)
        for t in 1:T
            pred = predict(rng, dyn(model), df, t, s, observations[t]; kwargs...)
            predicted[t] = pred
            s, ll = update(obs(model), df, t, pred, observations[t]; kwargs...)
            filtered[t] = s
            total_ll += ll
        end
        s
    end

    # Backward pass
    smoothed = let s = filtered[T]
        for t in (T - 1):-1:t_smooth
            s = backward_smooth(
                dyn(model), df, t, filtered[t], s; predicted=predicted[t + 1], kwargs...
            )
        end
        s
    end

    return smoothed, total_ll
end

## TWO-FILTER SMOOTHING ####################################################################

"""
    two_filter_smooth(filtered::AbstractVector, backward_lik::DiscreteLikelihood)

Combine forward filtered distribution with backward likelihood to get smoothed distribution.

Returns the normalized smoothed distribution γ_t(i) ∝ π_t(i) * β_t(i).

All computations are performed in log-space for numerical stability.
"""
function two_filter_smooth(filtered::AbstractVector, backward_lik::DiscreteLikelihood)
    log_filtered = log.(filtered)
    log_β = log_likelihoods(backward_lik)

    log_smoothed = log_filtered .+ log_β
    log_normalizer = logsumexp(log_smoothed)

    return exp.(log_smoothed .- log_normalizer)
end
