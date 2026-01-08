## FILTERING INTERFACE #####################################################################

export initialise, step, predict, update, filter

"""
    initialise([rng,] model, algo; kwargs...)

Propose an initial state distribution from the prior.

# Arguments
- `rng`: Random number generator (optional, defaults to `default_rng()`)
- `model`: The state space model (or its prior component)
- `algo`: The filtering algorithm

# Returns
The initial state distribution.
"""
function initialise end

"""
    step([rng,] model, algo, iter, state, observation; kwargs...)

Perform a combined predict and update step of the filtering algorithm.

# Arguments
- `rng`: Random number generator (optional)
- `model`: The state space model
- `algo`: The filtering algorithm
- `iter::Integer`: The current time step
- `state`: The current state distribution
- `observation`: The observation at time `iter`

# Returns
A tuple `(new_state, log_likelihood_increment)`.
"""
function step end

"""
    predict([rng,] dyn, algo, iter, state, observation; kwargs...)

Propagate the filtered distribution forward in time using the dynamics model.

# Arguments
- `rng`: Random number generator (optional)
- `dyn`: The latent dynamics model
- `algo`: The filtering algorithm
- `iter::Integer`: The current time step
- `state`: The filtered state distribution at time `iter-1`
- `observation`: The observation (may be used by some proposals)

# Returns
The predicted state distribution at time `iter`.
"""
function predict end

"""
    update(obs, algo, iter, state, observation; kwargs...)

Update the predicted distribution given an observation.

# Arguments
- `obs`: The observation process model
- `algo`: The filtering algorithm
- `iter::Integer`: The current time step
- `state`: The predicted state distribution
- `observation`: The observation at time `iter`

# Returns
A tuple `(filtered_state, log_likelihood_increment)`.
"""
function update end

function initialise(model, algo; kwargs...)
    return initialise(default_rng(), model, algo; kwargs...)
end

function predict(model, algo, step, filtered, observation; kwargs...)
    return predict(default_rng(), model, algo, step, filtered, observation; kwargs...)
end

## SMOOTHING INTERFACE #####################################################################

export smooth, backward_smooth, two_filter_smooth

@doc raw"""
    backward_smooth(dyn, algo, step, filtered, smoothed_next; predicted=nothing, kwargs...)

Perform a single backward smoothing step, computing the smoothed distribution at time `step`
given the filtered distribution at time `step` and the smoothed distribution at time `step+1`.

For efficiency, the predicted distribution `p(x_{t+1} | y_{1:t})` should be provided via the
`predicted` keyword argument if available from the forward pass. If omitted, it will be
recomputed internally, which duplicates work.

This implements the backward recursion of the forward-backward (Rauch-Tung-Striebel) smoother:

```math
p(x_t | y_{1:T}) \propto p(x_t | y_{1:t}) \int \frac{p(x_{t+1} | x_t) \, p(x_{t+1} | y_{1:T})}{p(x_{t+1} | y_{1:t})} \, dx_{t+1}
```

# Arguments
- `dyn`: The latent dynamics model
- `algo`: The filtering algorithm (determines the state representation)
- `step::Integer`: The time index t of the filtered state
- `filtered`: The filtered distribution ``p(x_t | y_{1:t})``
- `smoothed_next`: The smoothed distribution ``p(x_{t+1} | y_{1:T})``
- `predicted=nothing`: The predicted distribution ``p(x_{t+1} | y_{1:t})``. Should be provided
  if available; if `nothing`, it is recomputed from `filtered`.

# Returns
The smoothed distribution ``p(x_t | y_{1:T})``.

# Implementations
- **Linear Gaussian** (`LinearGaussianLatentDynamics`, `KalmanFilter`): Returns `MvNormal`
  using the RTS equations with smoothing gain ``G = \Sigma_{\text{filt}} A^\top \Sigma_{\text{pred}}^{-1}``

See also: [`two_filter_smooth`](@ref), [`smooth`](@ref)
"""
function backward_smooth end

@doc raw"""
    two_filter_smooth(filtered, backward_lik)

Combine a forward filtered distribution with a backward predictive likelihood to obtain
the smoothed distribution at a given time step.

The smoothed distribution is the (normalized) product:

```math
p(x_t | y_{1:T}) \propto p(x_t | y_{1:t}) \times p(y_{t+1:T} | x_t)
```

where:
- `filtered` represents the forward filtered distribution ``p(x_t | y_{1:t})``
- `backward_lik` represents the backward predictive likelihood ``p(y_{t+1:T} | x_t)``

Note that `backward_lik` is a likelihood (function of x), not a distribution over x.

# Arguments
- `filtered`: The filtered distribution ``p(x_t | y_{1:t})``
- `backward_lik`: The backward predictive likelihood ``p(y_{t+1:T} | x_t)``

# Returns
The smoothed distribution ``p(x_t | y_{1:T})``.

# Implementations
- **Linear Gaussian** (`MvNormal`, `InformationLikelihood`): Combines using the product
  of Gaussians formula in information form.

# Relation to `compute_marginal_predictive_likelihood`
Both functions take the same inputs but compute different quantities:
- `two_filter_smooth` returns the **distribution** ``p(x_t | y_{1:T})``
- `compute_marginal_predictive_likelihood` returns the **scalar** ``p(y_{t+1:T} | y_{1:t})``

See also: [`backward_smooth`](@ref), [`compute_marginal_predictive_likelihood`](@ref)
"""
function two_filter_smooth end

## BACKWARD LIKELIHOOD INTERFACE ###########################################################

export backward_initialise, backward_predict, backward_update

"""
    backward_initialise(rng, obs, algo, iter, observation; kwargs...)

Initialize the backward likelihood at the final time step T.

This creates an initial representation of p(y_T | x_T), the likelihood of the final
observation given the state at time T.

# Arguments
- `rng`: Random number generator
- `obs`: The observation process model
- `algo`: The backward predictor algorithm
- `iter::Integer`: The final time step T
- `observation`: The observation y_T at time T

# Returns
A representation of the likelihood p(y_T | x_T).

# Implementations
- **Linear Gaussian** (`LinearGaussianObservationProcess`, `BackwardInformationPredictor`):
  Returns `InformationLikelihood` with λ = H'R⁻¹(y-c) and Ω = H'R⁻¹H

See also: [`backward_predict`](@ref), [`backward_update`](@ref)
"""
function backward_initialise end

"""
    backward_predict(rng, dyn, algo, iter, state; kwargs...)

Perform a backward prediction step, propagating the likelihood backward through the dynamics.

Given a representation of p(y_{t+1:T} | x_{t+1}), compute p(y_{t+1:T} | x_t) by
marginalizing over the transition:

    p(y_{t+1:T} | x_t) = ∫ p(x_{t+1} | x_t) p(y_{t+1:T} | x_{t+1}) dx_{t+1}

# Arguments
- `rng`: Random number generator
- `dyn`: The latent dynamics model
- `algo`: The backward predictor algorithm
- `iter::Integer`: The time step t (predicting from t to t+1)
- `state`: The backward likelihood p(y_{t+1:T} | x_{t+1})

# Returns
The backward likelihood p(y_{t+1:T} | x_t).

# Implementations
- **Linear Gaussian** (`LinearGaussianLatentDynamics`, `BackwardInformationPredictor`):
  Updates `InformationLikelihood` using the backward information filter equations.

See also: [`backward_initialise`](@ref), [`backward_update`](@ref)
"""
function backward_predict end

"""
    backward_update(obs, algo, iter, state, observation; kwargs...)

Incorporate an observation into the backward likelihood.

Given p(y_{t+1:T} | x_t), incorporate observation y_t to obtain p(y_{t:T} | x_t):

    p(y_{t:T} | x_t) = p(y_t | x_t) × p(y_{t+1:T} | x_t)

# Arguments
- `obs`: The observation process model
- `algo`: The backward predictor algorithm
- `iter::Integer`: The time step t
- `state`: The backward likelihood p(y_{t+1:T} | x_t)
- `observation`: The observation y_t at time t

# Returns
The updated backward likelihood p(y_{t:T} | x_t).

# Implementations
- **Linear Gaussian** (`LinearGaussianObservationProcess`, `BackwardInformationPredictor`):
  Adds observation information: λ += H'R⁻¹(y-c), Ω += H'R⁻¹H

See also: [`backward_initialise`](@ref), [`backward_predict`](@ref)
"""
function backward_update end
