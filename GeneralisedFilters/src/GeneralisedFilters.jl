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

## FILTERING BASE ##########################################################################

abstract type AbstractFilter <: AbstractSampler end
abstract type AbstractBatchFilter <: AbstractFilter end

"""
    initialise([rng,] model, algo; kwargs...)

Propose an initial state distribution.
"""
function initialise end

"""
    step([rng,] model, algo, iter, state, observation; kwargs...)

Perform a combined predict and update call of the filtering on the state.
"""
function step end

"""
    predict([rng,] model, algo, iter, filtered; kwargs...)

Propagate the filtered distribution forward in time.
"""
function predict end

"""
    update(model, algo, iter, proposed, observation; kwargs...)

Update beliefs on the propagated distribution given an observation.
"""
function update end

function initialise(model, algo; kwargs...)
    return initialise(default_rng(), model, algo; kwargs...)
end

function predict(model, algo, step, filtered, observation; kwargs...)
    return predict(default_rng(), model, algo, step, filtered; kwargs...)
end

function filter(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    algo::AbstractFilter,
    observations::AbstractVector;
    callback::Union{AbstractCallback,Nothing}=nothing,
    kwargs...,
)
    state = initialise(rng, model, algo; kwargs...)
    isnothing(callback) || callback(model, algo, state, observations, PostInit; kwargs...)

    log_evidence = initialise_log_evidence(algo, model)

    for t in eachindex(observations)
        state, ll_increment = step(
            rng, model, algo, t, state, observations[t]; callback, kwargs...
        )
        log_evidence += ll_increment
    end

    return state, log_evidence
end

function initialise_log_evidence(::AbstractFilter, model::AbstractStateSpaceModel)
    return 0
end

# function initialise_log_evidence(alg::AbstractBatchFilter, model::AbstractStateSpaceModel)
#     return CUDA.zeros(SSMProblems.arithmetic_type(model), alg.batch_size)
# end

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
    callback::Union{AbstractCallback,Nothing}=nothing,
    kwargs...,
)
    state = predict(rng, model, algo, iter, state, observation; kwargs...)
    isnothing(callback) ||
        callback(model, algo, iter, state, observation, PostPredict; kwargs...)

    state, ll_increment = update(model, algo, iter, state, observation; kwargs...)
    isnothing(callback) ||
        callback(model, algo, iter, state, observation, PostUpdate; kwargs...)

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
