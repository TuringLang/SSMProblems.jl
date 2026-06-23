"""Storage and callback methods for recording historic filtering information."""

export AbstractCallback, CallbackTrigger
export PostInit, PostResample, PostPredict, PostUpdate
export PostInitCallback, PostResampleCallback, PostPredictCallback, PostUpdateCallback
export DenseAncestorCallback, AncestorCallback

## ABSTRACT CALLBACK SYSTEM ################################################################

abstract type AbstractCallback end

abstract type CallbackTrigger end

const CallbackType = Union{Nothing,<:AbstractCallback}

struct PostInitCallback <: CallbackTrigger end
const PostInit = PostInitCallback()
function (c::CallbackType)(model, filter, state, data, ::PostInitCallback; kwargs...)
    return nothing
end

struct PostResampleCallback <: CallbackTrigger end
const PostResample = PostResampleCallback()
function (c::CallbackType)(
    model, filter, step, state, data, ::PostResampleCallback; kwargs...
)
    return nothing
end

struct PostPredictCallback <: CallbackTrigger end
const PostPredict = PostPredictCallback()
function (c::CallbackType)(
    model, filter, step, state, data, ::PostPredictCallback; kwargs...
)
    return nothing
end

struct PostUpdateCallback <: CallbackTrigger end
const PostUpdate = PostUpdateCallback()
function (c::CallbackType)(
    model, filter, step, state, data, ::PostUpdateCallback; kwargs...
)
    return nothing
end

## DENSE ANCESTOR CALLBACK #################################################################

"""
    DenseAncestorCallback

A callback for dense ancestry storage, which fills a `DenseParticleContainer`.
"""
mutable struct DenseAncestorCallback <: AbstractCallback
    container
end

function (c::DenseAncestorCallback)(
    model, filter, state, data, ::PostInitCallback; kwargs...
)
    particles = state.particles
    # Defer construction: the time-0 particles live here but we don't yet know the
    # subsequent-state type. Using the one-argument constructor assumes T == T0; for
    # filters where the two differ (e.g. RBPF with typed inner distributions), the user
    # would need a filter-specific callback or explicit type argument.
    c.container = DenseParticleContainer(deepcopy(getfield.(particles, :state)))
    return nothing
end

function (c::DenseAncestorCallback)(
    model, filter, step, state, data, ::PostUpdateCallback; kwargs...
)
    particles = state.particles
    push!(
        c.container,
        deepcopy(getfield.(particles, :state)),
        deepcopy(Float64.(getfield.(particles, :log_w))),
        deepcopy(getfield.(particles, :ancestor)),
    )
    return nothing
end

## ANCESTOR STORAGE CALLBACK ###############################################################

"""
    AncestorCallback

A callback for sparse ancestry storage, which preallocates and returns a populated 
`ParticleTree` object.
"""
mutable struct AncestorCallback <: AbstractCallback
    tree
end

function (c::AncestorCallback)(model, filter, state, data, ::PostInitCallback; kwargs...)
    N = num_particles(filter)
    c.tree = ParticleTree(
        getfield.(state.particles, :state), max(N, floor(Int64, N * log(N)))
    )
    return nothing
end

function (c::AncestorCallback)(
    model, filter, step, state, data, ::PostPredictCallback; kwargs...
)
    particles = state.particles
    ancestors = getfield.(particles, :ancestor)
    prune!(c.tree, get_offspring(ancestors))
    insert!(c.tree, getfield.(particles, :state), ancestors)
    return nothing
end

"""
    ResamplerCallback

A callback which follows the resampling indices over the filtering algorithm. This is more
of a debug tool and visualizer for various resapmling algorithms.
"""
mutable struct ResamplerCallback <: AbstractCallback
    tree
end

function (c::ResamplerCallback)(model, filter, state, data, ::PostInitCallback; kwargs...)
    N = num_particles(filter)
    c.tree = ParticleTree(collect(1:N), max(N, floor(Int64, N * log(N))))
    return nothing
end

function (c::ResamplerCallback)(
    model, filter, step, state, data, ::PostResampleCallback; kwargs...
)
    N = num_particles(filter)
    prune!(c.tree, get_offspring(state.ancestors))
    insert!(c.tree, collect(1:N), state.ancestors)
    return nothing
end
