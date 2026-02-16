module GeneralisedFilters

using AbstractMCMC: AbstractMCMC, AbstractSampler
import Distributions: MvNormal, params
import Random: AbstractRNG, default_rng, rand
import SSMProblems: prior, dyn, obs
using OffsetArrays
using SSMProblems
using StatsBase

# TODO: heavy modules—move to extension
using CUDA

# Filtering utilities
include("callbacks.jl")
include("containers.jl")
include("resamplers.jl")

## FILTERING BASE ##########################################################################

abstract type AbstractFilter <: AbstractSampler end
abstract type AbstractBackwardPredictor <: AbstractSampler end

# Abstract interface definitions (filtering, smoothing, backward likelihood)
include("algorithms/interface.jl")

"""
    filter([rng,] model, algo, observations; callback=nothing, kwargs...)

Run a filtering algorithm on a state-space model.

Performs sequential Bayesian inference by iterating through observations,
calling [`predict`](@ref) and [`update`](@ref) at each time step.

# Arguments
- `rng::AbstractRNG`: Random number generator (optional, defaults to `default_rng()`)
- `model::AbstractStateSpaceModel`: The state-space model to filter
- `algo::AbstractFilter`: The filtering algorithm (e.g., `KalmanFilter()`, `BootstrapFilter(N)`)
- `observations::AbstractVector`: Vector of observations y₁:ₜ

# Keyword Arguments
- `callback`: Optional callback for recording intermediate states
- `kwargs...`: Additional arguments passed to model parameter functions

# Returns
A tuple `(state, log_likelihood)` where:
- `state`: The final filtered state (algorithm-dependent type)
- `log_likelihood`: The total log-marginal likelihood log p(y₁:ₜ)

# Example
```julia
model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
state, ll = filter(model, KalmanFilter(), observations)
```

See also: [`predict`](@ref), [`update`](@ref), [`step`](@ref), [`smooth`](@ref)
"""
function filter(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    algo::AbstractFilter,
    observations::AbstractVector;
    callback::CallbackType=nothing,
    kwargs...,
)
    # draw from the prior
    init_state = initialise(rng, prior(model), algo; kwargs...)
    callback(model, algo, init_state, observations, PostInit; kwargs...)

    # iterations starts here for type stability
    state, log_evidence = step(
        rng, model, algo, 1, init_state, observations[1]; callback, kwargs...
    )

    # subsequent iteration
    for t in 2:length(observations)
        state, ll_increment = step(
            rng, model, algo, t, state, observations[t]; callback, kwargs...
        )
        log_evidence += ll_increment
    end

    return state, log_evidence
end

function filter(
    model::AbstractStateSpaceModel,
    algo::AbstractFilter,
    observations::AbstractVector;
    kwargs...,
)
    return filter(default_rng(), model, algo, observations; kwargs...)
end

function step(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    algo::AbstractFilter,
    iter::Integer,
    state,
    observation;
    kwargs...,
)
    # generalised to fit analytical filters
    return move(rng, model, algo, iter, state, observation; kwargs...)
end
function step(
    model::AbstractStateSpaceModel,
    algo::AbstractFilter,
    iter::Integer,
    state,
    observation;
    kwargs...,
)
    return step(default_rng(), model, algo, iter, state, observation; kwargs...)
end

function move(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    algo::AbstractFilter,
    iter::Integer,
    state,
    observation;
    callback::CallbackType=nothing,
    kwargs...,
)
    state = predict(rng, dyn(model), algo, iter, state, observation; kwargs...)
    callback(model, algo, iter, state, observation, PostPredict; kwargs...)

    state, ll_increment = update(obs(model), algo, iter, state, observation; kwargs...)
    callback(model, algo, iter, state, observation, PostUpdate; kwargs...)

    return state, ll_increment
end

## SMOOTHING BASE ##########################################################################

abstract type AbstractSmoother <: AbstractSampler end

# Model types
include("models/linear_gaussian.jl")
include("models/discrete.jl")
include("models/hierarchical.jl")

# Filtering/smoothing algorithms
include("algorithms/particles.jl")
include("algorithms/kalman.jl")
include("algorithms/srkf.jl")
include("algorithms/kalman_gradient.jl")
include("algorithms/forward.jl")
include("algorithms/rbpf.jl")

include("ancestor_sampling.jl")

# Conditional SMC (particle Gibbs trajectory sampling)
include("algorithms/csmc.jl")

# Unit-testing helper module
include("GFTest/GFTest.jl")

end
