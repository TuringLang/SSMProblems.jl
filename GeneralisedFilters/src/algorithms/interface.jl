## FILTERING INTERFACE #####################################################################

export initialise, step, predict, update, filter

"""
    initialise([rng,] prior, algo; ref_state=nothing)

Construct the initial state representation from the prior for a filtering algorithm.
"""
function initialise end

"""
    predict([rng,] dyn, algo, t, state, y; ref_state=nothing)

Propagate the state representation forward through the dynamics.
"""
function predict end

"""
    update(obs, algo, t, state, y)

Incorporate observation `y` into the predicted state, returning
`(filtered_state, ll_increment)`.
"""
function update end

"""
    step([rng,] model, algo, t, state, y; ref_state=nothing)

Combined predict-and-update step, returning `(new_state, ll_increment)`.
"""
function step end

initialise(prior, algo; kwargs...) = initialise(default_rng(), prior, algo; kwargs...)
function predict(dyn, algo, t, state, y; kwargs...)
    return predict(default_rng(), dyn, algo, t, state, y; kwargs...)
end

## SMOOTHING INTERFACE #####################################################################

export smooth, two_filter_smooth

"""
    smooth([rng,] model, algo, ys; t_smooth=1)

Run a forward-backward smoothing pass, returning `(smoothed_state, total_ll)` where the
smoothed state is `p(x_{t_smooth} | y_{1:T})`.
"""
function smooth end

"""
    two_filter_smooth(filtered, backward_lik)

Combine a forward filtered distribution `p(x_t | y_{1:t})` with a backward predictive
likelihood `p(y_{t+1:T} | x_t)` to obtain the smoothed distribution `p(x_t | y_{1:T})`.
"""
function two_filter_smooth end

## BACKWARD LIKELIHOOD INTERFACE ###########################################################

export backward_initialise, backward_predict, backward_update

"""
    backward_initialise(algo, obs, ...)

Initialise the backward predictive likelihood at the final time step.
"""
function backward_initialise end

"""
    backward_predict(algo, lik, dyn)

Propagate the backward predictive likelihood through the dynamics.
"""
function backward_predict end

"""
    backward_update(algo, lik, obs, ...)

Incorporate an observation into the backward predictive likelihood.
"""
function backward_update end
