import AbstractMCMC: AbstractSampler

## FILTERING ###############################################################################

abstract type AbstractFilter <: AbstractSampler end

"""
    predict([rng,] model, alg, step, states, [extra])

propagate the filtered states forward in time.
"""
function predict end

function predict(model, alg, step, states; kwargs...)
    return predict(default_rng(), model, alg, step, states; kwargs...)
end

"""
    update(model, alg, step, states, data, [extra])

update beliefs on the propagated states.
"""
function update end

"""
    initialise([rng,] model, alg, [extra])

propose an initial state distribution.
"""
function initialise end

function initialise(model, alg; kwargs...)
    return initialise(default_rng(), model, alg; kwargs...)
end

function filter(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    alg::AbstractFilter,
    observations::AbstractVector;
    callback=nothing,
    kwargs...,
)
    states = initialise(rng, model, alg; kwargs...)
    log_evidence = zero(eltype(model))

    for t in eachindex(observations)
        states, log_marginal = step(
            rng, model, alg, t, states, observations[t]; callback, kwargs...
        )
        log_evidence += log_marginal
        isnothing(callback) || callback(model, alg, t, states, observations; kwargs...)
    end
    
    return states, log_evidence
end

function filter(
    model::AbstractStateSpaceModel,
    alg::AbstractFilter,
    observations::AbstractVector;
    kwargs...
)
    return filter(default_rng(), model, alg, observations; kwargs...)
end

function step(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    alg::AbstractFilter,
    t::Integer,
    state,
    observation;
    kwargs...,
)
    proposed_state = predict(rng, model, alg, t, state; kwargs...)
    filtered_state, ll = update(
        model, alg, t, proposed_state, observation; kwargs...
    )

    return filtered_state, ll
end