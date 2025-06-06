module GeneralisedFilters

using AbstractMCMC: AbstractMCMC, AbstractSampler
import Distributions: MvNormal
import Random: AbstractRNG, default_rng, rand
using GaussianDistributions: pairs, Gaussian
using OffsetArrays
using SSMProblems
using StatsBase

# TODO: heavy modules—move to extension
using CUDA
using NNlib

# Filtering utilities
include("callbacks.jl")
include("containers.jl")
include("resamplers.jl")

# Batching utilities
include("batching/batched_CUDA.jl")

## FILTERING BASE ##########################################################################

abstract type AbstractFilter <: AbstractSampler end
abstract type AbstractBatchFilter <: AbstractFilter end

"""
    initialise([rng,] model, alg; kwargs...)

Propose an initial state distribution.
"""
function initialise end

"""
    step([rng,] model, alg, iter, state, observation; kwargs...)

Perform a combined predict and update call of the filtering on the state.
"""
function step end

"""
    predict([rng,] model, alg, iter, filtered; kwargs...)

Propagate the filtered distribution forward in time.
"""
function predict end

"""
    update(model, alg, iter, proposed, observation; kwargs...)

Update beliefs on the propagated distribution given an observation.
"""
function update end

function initialise(model, alg; kwargs...)
    return initialise(default_rng(), model, alg; kwargs...)
end

function predict(model, alg, step, filtered, observation; kwargs...)
    return predict(default_rng(), model, alg, step, filtered; kwargs...)
end

function filter(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    alg::AbstractFilter,
    observations::AbstractVector;
    callback::Union{AbstractCallback,Nothing}=nothing,
    kwargs...,
)
    state = initialise(rng, model, alg; kwargs...)
    isnothing(callback) || callback(model, alg, state, observations, PostInit; kwargs...)

    log_evidence = initialise_log_evidence(alg, model)

    for t in eachindex(observations)
        state, ll_increment = step(
            rng, model, alg, t, state, observations[t]; callback, kwargs...
        )
        log_evidence += ll_increment
    end

    return state, log_evidence
end

function initialise_log_evidence(::AbstractFilter, model::AbstractStateSpaceModel)
    return zero(SSMProblems.arithmetic_type(model))
end

function initialise_log_evidence(alg::AbstractBatchFilter, model::AbstractStateSpaceModel)
    return CUDA.zeros(SSMProblems.arithmetic_type(model), alg.batch_size)
end

function filter(
    model::AbstractStateSpaceModel,
    alg::AbstractFilter,
    observations::AbstractVector;
    kwargs...,
)
    return filter(default_rng(), model, alg, observations; kwargs...)
end

function step(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    alg::AbstractFilter,
    iter::Integer,
    state,
    observation;
    callback::Union{AbstractCallback,Nothing}=nothing,
    kwargs...,
)
    state = predict(rng, model, alg, iter, state, observation; kwargs...)
    isnothing(callback) ||
        callback(model, alg, iter, state, observation, PostPredict; kwargs...)

    state, ll_increment = update(model, alg, iter, state, observation; kwargs...)
    isnothing(callback) ||
        callback(model, alg, iter, state, observation, PostUpdate; kwargs...)

    return state, ll_increment
end

## SMOOTHING BASE ##########################################################################

abstract type AbstractSmoother <: AbstractSampler end

# function smooth end
# function backward end

# Model types
include("models/linear_gaussian.jl")
include("models/discrete.jl")
include("models/hierarchical.jl")

# Filtering/smoothing algorithms
include("algorithms/particles.jl")
include("algorithms/kalman.jl")
include("algorithms/forward.jl")
include("algorithms/rbpf.jl")

# Unit-testing helper module
include("GFTest/GFTest.jl")

end
