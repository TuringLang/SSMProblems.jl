module GeneralisedFilters

using AbstractMCMC: AbstractMCMC, AbstractSampler
using Distributions: Distributions, MvNormal, Categorical, logpdf
using LinearAlgebra
using Random: Random, AbstractRNG, default_rng, randn
import Random: rand
using StaticArrays
using Statistics: Statistics, mean, cov
using StatsBase
using LogExpFunctions: LogExpFunctions, logsumexp, softmax
using DataStructures: DataStructures

## ALGORITHM TYPE HIERARCHY ################################################################

abstract type AbstractFilter <: AbstractSampler end
abstract type AbstractParticleFilter <: AbstractFilter end
abstract type AbstractSmoother <: AbstractSampler end
abstract type AbstractBackwardPredictor <: AbstractSampler end

## CORE TYPES AND CONTAINERS ###############################################################

include("gaussian.jl")
include("containers.jl")
include("resamplers.jl")

## MODEL LAYER #############################################################################

include("models/interface.jl")
include("models/atoms.jl")
include("models/hierarchical.jl")

## ACTIVITY ################################################################################

include("activity.jl")

## KERNELS #################################################################################

include("kernels/kalman.jl")
include("kernels/kalman_adjoint.jl")

## FILTERING/SMOOTHING #####################################################################

include("algorithms/interface.jl")

"""
    filter([rng,] model, algo, ys; ref_state=nothing)

Run a filtering algorithm over nonempty observations `ys`, returning `(final_state, total_ll)`.
An empty observation sequence raises an `ArgumentError`.
"""
function filter(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    algo::AbstractFilter,
    ys::AbstractVector;
    ref_state=nothing,
)
    _validate_observations(model, ys)
    isempty(ys) && throw(ArgumentError("filter requires nonempty observations"))
    init_state = initialise(rng, SSMProblems.prior(model), algo; ref_state)

    # First iteration peeled out for type stability.
    state, log_evidence = step(rng, model, algo, 1, init_state, ys[1]; ref_state)
    for t in 2:length(ys)
        state, ll_increment = step(rng, model, algo, t, state, ys[t]; ref_state)
        log_evidence += ll_increment
    end

    return state, log_evidence
end
function filter(
    model::AbstractStateSpaceModel, algo::AbstractFilter, ys::AbstractVector; kwargs...
)
    return filter(default_rng(), model, algo, ys; kwargs...)
end

function step(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    algo::AbstractFilter,
    t::Integer,
    state,
    y;
    ref_state=nothing,
)
    return move(rng, model, algo, t, state, y; ref_state)
end
function step(
    model::AbstractStateSpaceModel, algo::AbstractFilter, t::Integer, state, y; kwargs...
)
    return step(default_rng(), model, algo, t, state, y; kwargs...)
end

function move(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    algo::AbstractFilter,
    t::Integer,
    state,
    y;
    ref_state=nothing,
)
    state = predict(rng, SSMProblems.dyn(model), algo, t, state, y; ref_state)
    state, ll_increment = update(SSMProblems.obs(model), algo, t, state, y)
    return state, ll_increment
end

## ALGORITHMS ##############################################################################

include("algorithms/kalman.jl")
include("algorithms/srkf.jl")
include("algorithms/forward.jl")
include("algorithms/particles.jl")
include("algorithms/rbpf.jl")
include("ancestor_sampling.jl")
include("algorithms/csmc.jl")

include("integrations/conditional_logdensity.jl")
include("integrations/logdensity.jl")
include("integrations/particle_gibbs.jl")
include("integrations/ssm_trajectory.jl")

## TEST UTILITIES ##########################################################################

include("GFTest/GFTest.jl")

end
