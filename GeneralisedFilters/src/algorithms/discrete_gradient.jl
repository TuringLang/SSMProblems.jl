export DiscretePredictGradientCache, DiscreteGradientCache
export predict_with_cache, update_with_cache
export gradient_P, gradient_log_emission

"""
    DiscretePredictGradientCache

Cache for the discrete/HMM predict step:
    s_t = P_t' * f_{t-1}

Stores the previous filtered distribution and transition matrix so that the backward pass can
propagate adjoints and accumulate transition gradients.
"""
struct DiscretePredictGradientCache{FT,PT}
    filtered_prev::FT
    P::PT
end

"""
    DiscreteGradientCache

Cache for the discrete/HMM update step:
    u_t = b_t .* s_t
    c_t = sum(u_t)
    f_t = u_t / c_t

Used to backpropagate through the normalized filtering recursion and compute emission
likelihood gradients.
"""
struct DiscreteGradientCache{PT,FT,BT,CT}
    predicted::PT
    filtered::FT
    emission_probs::BT
    normalizer::CT
end

"""
    predict_with_cache(rng, dyn, algo::DiscreteFilter, iter, states, observation; kwargs...)

Perform a discrete predict step and return `(predicted, cache)`, where `cache` stores
`f_{t-1}` and `P_t` for the backward pass.
"""
function predict_with_cache(
    rng::AbstractRNG,
    dyn::DiscreteLatentDynamics,
    algo::DiscreteFilter,
    iter::Integer,
    states::AbstractVector,
    observation=nothing;
    kwargs...,
)
    P = calc_P(dyn, iter; kwargs...)
    predicted = (states' * P)'
    cache = DiscretePredictGradientCache(states, P)
    return predicted, cache
end

function predict_with_cache(
    dyn::DiscreteLatentDynamics,
    algo::DiscreteFilter,
    iter::Integer,
    states::AbstractVector,
    observation=nothing;
    kwargs...,
)
    return predict_with_cache(default_rng(), dyn, algo, iter, states, observation; kwargs...)
end

"""
    update_with_cache(obs, algo::DiscreteFilter, iter, state, observation; kwargs...)

Perform a discrete update step and return `(filtered, log_likelihood, cache)`.
"""
function update_with_cache(
    obs::ObservationProcess,
    algo::DiscreteFilter,
    iter::Integer,
    state::AbstractVector,
    observation;
    kwargs...,
)
    emission_probs = map(
        x -> exp(SSMProblems.logdensity(obs, iter, x, observation; kwargs...)),
        eachindex(state),
    )

    unnormalized = emission_probs .* state
    normalizer = sum(unnormalized)
    filtered = unnormalized / normalizer
    ll = log(normalizer)

    cache = DiscreteGradientCache(state, filtered, emission_probs, normalizer)
    return filtered, ll, cache
end

## BACKWARD GRADIENT PROPAGATION ##############################################################

"""
    backward_gradient_update(∂filtered, cache::DiscreteGradientCache)

Backpropagate through the normalized discrete update:
    u_t = b_t .* s_t, c_t = sum(u_t), f_t = u_t / c_t

Implements NLL gradients (local term `-log(c_t)` included).

Returns `(∂predicted, ∂emission_probs)`.
"""
function backward_gradient_update(∂filtered::AbstractVector, cache::DiscreteGradientCache)
    filtered = cache.filtered
    predicted = cache.predicted
    emission_probs = cache.emission_probs
    c = cache.normalizer

    inv_c = one(c) / c
    # Jacobian-vector product for normalization f = u / sum(u), plus local NLL term -log(c)
    dot_term = dot(∂filtered, filtered)
    ∂unnormalized = inv_c .* (∂filtered .- dot_term) .- inv_c

    ∂predicted = ∂unnormalized .* emission_probs
    ∂emission_probs = ∂unnormalized .* predicted

    return ∂predicted, ∂emission_probs
end

"""
    backward_gradient_predict(∂predicted, P)
    backward_gradient_predict(∂predicted, cache::DiscretePredictGradientCache)

Backpropagate through the discrete predict step:
    s_t = P_t' * f_{t-1}
"""
function backward_gradient_predict(∂predicted::AbstractVector, P::AbstractMatrix)
    return P * ∂predicted
end

function backward_gradient_predict(
    ∂predicted::AbstractVector, cache::DiscretePredictGradientCache
)
    return backward_gradient_predict(∂predicted, cache.P)
end

## PARAMETER GRADIENTS ########################################################################

"""
    gradient_P(∂predicted, filtered_prev)
    gradient_P(∂predicted, cache::DiscretePredictGradientCache)

Gradient of NLL with respect to transition matrix `P` in:
    s_t = P_t' * f_{t-1}
"""
function gradient_P(∂predicted::AbstractVector, filtered_prev::AbstractVector)
    return filtered_prev * ∂predicted'
end

function gradient_P(∂predicted::AbstractVector, cache::DiscretePredictGradientCache)
    return gradient_P(∂predicted, cache.filtered_prev)
end

"""
    gradient_log_emission(∂emission_probs, emission_probs)
    gradient_log_emission(∂emission_probs, cache::DiscreteGradientCache)

Map gradients from emission probabilities `b_t` to log-emission likelihoods `log b_t` using:
    ∂L/∂log b_t = (∂L/∂b_t) ⊙ b_t
"""
function gradient_log_emission(
    ∂emission_probs::AbstractVector, emission_probs::AbstractVector
)
    return ∂emission_probs .* emission_probs
end

function gradient_log_emission(
    ∂emission_probs::AbstractVector, cache::DiscreteGradientCache
)
    return gradient_log_emission(∂emission_probs, cache.emission_probs)
end
