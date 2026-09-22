export DiscreteFilter, DF
export BackwardDiscretePredictor
export DiscreteSmoother

"""
    DiscreteFilter <: AbstractFilter

Forward filtering algorithm for discrete (finite) state-space models. Computes the filtered
distribution `π_t(i) = p(x_t = i | y_{1:t})`.
"""
struct DiscreteFilter <: AbstractFilter end
const DF = DiscreteFilter

function initialise(
    ::AbstractRNG, prior::DiscretePrior, ::DiscreteFilter; ref_state=nothing
)
    return prior.α0
end

function predict(
    ::AbstractRNG,
    dyn,
    ::DiscreteFilter,
    t::Integer,
    π::AbstractVector,
    y;
    ref_state=nothing,
)
    return vec(π' * resolve(dyn, (; t)).P)
end

function update(obs, ::DiscreteFilter, t::Integer, π::AbstractVector, y)
    o = resolve(obs, (; t))
    log_weights = map(i -> log(π[i]) + logdensity(o, t, i, y), eachindex(π))
    ll = logsumexp(log_weights)
    # Impossible observations have zero evidence; avoid manufacturing NaN beliefs.
    ll == -Inf && return zero.(π), ll
    return exp.(log_weights .- ll), ll
end

"""
    marginal_loglikelihood(model, ::DiscreteFilter, ys)

Evaluate a finite-state model's deterministic forward likelihood. Emission weights are
normalised in log space; impossible observations return `-Inf`.
"""
function marginal_loglikelihood(
    model::StateSpaceModel, af::DiscreteFilter, ys::AbstractVector
)
    return last(filter(model, af, ys))
end

## BACKWARD DISCRETE PREDICTOR #############################################################

"""
    BackwardDiscretePredictor <: AbstractBackwardPredictor

Recursively computes the backward likelihood `β_t(i) = p(y_{t:T} | x_t = i)` for discrete
state-space models, in log-space for numerical stability.
"""
struct BackwardDiscretePredictor <: AbstractBackwardPredictor end

function backward_initialise(
    ::BackwardDiscretePredictor, obs, t::Integer, y, num_states::Integer
)
    o = resolve(obs, (; t))
    log_β = map(i -> logdensity(o, t, i, y), 1:num_states)
    return DiscreteLikelihood(log_β)
end

function backward_predict(
    ::BackwardDiscretePredictor, lik::DiscreteLikelihood, d::DiscreteDynamics
)
    log_β_next = log_likelihoods(lik)
    P = d.P
    K = length(log_β_next)
    log_β = map(1:K) do i
        return logsumexp(log.(P[i, :]) .+ log_β_next)
    end
    return DiscreteLikelihood(log_β)
end

function backward_update(
    ::BackwardDiscretePredictor, lik::DiscreteLikelihood, obs, t::Integer, y
)
    log_β = log_likelihoods(lik)
    K = length(log_β)
    o = resolve(obs, (; t))
    log_emission = map(i -> logdensity(o, t, i, y), 1:K)
    return DiscreteLikelihood(log_β .+ log_emission)
end

## DISCRETE SMOOTHER #######################################################################

"""
    DiscreteSmoother <: AbstractSmoother

Forward-backward smoother for discrete state-space models.
"""
struct DiscreteSmoother <: AbstractSmoother end

function _discrete_backward_step(
    d::DiscreteDynamics,
    filtered::AbstractVector,
    smoothed_next::AbstractVector,
    predicted::AbstractVector,
)
    P = d.P
    K = length(filtered)
    return map(1:K) do i
        correction = sum(1:K) do j
            # An unreachable state contributes zero, including the 0/0 case.
            if iszero(predicted[j])
                zero(smoothed_next[j])
            else
                P[i, j] * smoothed_next[j] / predicted[j]
            end
        end
        return filtered[i] * correction
    end
end

function smooth(
    rng::AbstractRNG,
    model::StateSpaceModel,
    ::DiscreteSmoother,
    ys::AbstractVector;
    t_smooth=1,
)
    _validate_observations(model, ys)
    1 <= t_smooth <= length(ys) ||
        throw(ArgumentError("smoothing time must lie in 1:length(ys)"))
    T = length(ys)
    df = DiscreteFilter()

    filtered = Vector{Vector{Float64}}(undef, T)
    predicted = Vector{Vector{Float64}}(undef, T)

    total_ll = 0.0
    state = let s = initialise(rng, model.prior, df)
        for t in 1:T
            pred = predict(rng, model.dyn, df, t, s, ys[t])
            predicted[t] = pred
            s, ll = update(model.obs, df, t, pred, ys[t])
            filtered[t] = s
            total_ll += ll
        end
        s
    end

    smoothed = let s = filtered[T]
        for t in (T - 1):-1:t_smooth
            # Atom index t+1 parameterises the transition x_t → x_{t+1}.
            d = resolve(model.dyn, (; t=t + 1))
            s = _discrete_backward_step(d, filtered[t], s, predicted[t + 1])
        end
        s
    end

    return smoothed, total_ll
end

## TWO-FILTER SMOOTHING ####################################################################

function two_filter_smooth(filtered::AbstractVector, backward_lik::DiscreteLikelihood)
    log_filtered = log.(filtered)
    log_β = log_likelihoods(backward_lik)
    log_smoothed = log_filtered .+ log_β
    log_normaliser = logsumexp(log_smoothed)
    return exp.(log_smoothed .- log_normaliser)
end

function compute_marginal_predictive_likelihood(
    forward::AbstractVector, backward::DiscreteLikelihood
)
    return logsumexp(log.(forward) .+ log_likelihoods(backward))
end
