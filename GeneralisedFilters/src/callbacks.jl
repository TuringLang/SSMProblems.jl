"""Storage and callback methods for recording historic filtering information."""

using DataStructures: Stack
using Random: rand

export AbstractCallback, CallbackTrigger
export PostInit, PostResample, PostPredict, PostUpdate
export PostInitCallback, PostResampleCallback, PostPredictCallback, PostUpdateCallback
export DenseParticleContainer, ParticleTree
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

## DENSE PARTICLE STORAGE ##################################################################

struct DenseParticleContainer{T,WT}
    particles::OffsetVector{Vector{T},Vector{Vector{T}}}
    weights::Vector{Vector{WT}}
    ancestors::Vector{Vector{Int}}
end

function get_ancestry(container::DenseParticleContainer{T}, i::Integer) where {T}
    a = i
    v = Vector{T}(undef, length(container.particles))
    ancestry = OffsetVector(v, -1)
    for t in length(container.ancestors):-1:1
        ancestry[t] = container.particles[t][a]
        a = container.ancestors[t][a]
    end
    ancestry[0] = container.particles[0][a]
    return ancestry
end

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
    c.container = DenseParticleContainer(
        OffsetVector([deepcopy(getfield.(particles, :state))], -1),
        Vector{Float64}[],
        Vector{Int}[],
    )
    return nothing
end

function (c::DenseAncestorCallback)(
    model, filter, step, state, data, ::PostUpdateCallback; kwargs...
)
    particles = state.particles
    push!(c.container.particles, deepcopy(getfield.(particles, :state)))
    push!(c.container.weights, deepcopy(getfield.(particles, :log_w)))
    push!(c.container.ancestors, deepcopy(getfield.(particles, :ancestor)))
    return nothing
end

## SPARSE PARTICLE STORAGE #################################################################

append_to_stack!(s::Stack, a::AbstractVector) = map(x -> push!(s, x), a)

"""
    ParticleTree

A sparse container for particle ancestry, which tracks the lineage of the filtered draws.

# Reference

Jacob, P., Murray L., & Rubenthaler S. (2015). Path storage in the particle 
filter [doi:10.1007/s11222-013-9445-x](https://dx.doi.org/10.1007/s11222-013-9445-x)
"""
struct ParticleTree{T}
    states::Vector{T}
    parents::Vector{Int64}
    leaves::Vector{Int64}
    offspring::Vector{Int64}
    free_indices::Stack{Int64}

    function ParticleTree(states::Vector{T}, M::Integer) where {T}
        nodes = Vector{T}(undef, M)
        initial_free_indices = Stack{Int64}()
        append_to_stack!(initial_free_indices, M:-1:(length(states) + 1))
        @inbounds nodes[1:length(states)] = states
        return new{T}(
            nodes, zeros(Int64, M), 1:length(states), zeros(Int64, M), initial_free_indices
        )
    end
end

Base.length(tree::ParticleTree) = length(tree.states)
Base.keys(tree::ParticleTree) = LinearIndices(tree.states)

function prune!(tree::ParticleTree, offspring::Vector{Int64})
    # insert new offspring counts
    setindex!(tree.offspring, offspring, tree.leaves)

    # update each branch
    @inbounds for i in eachindex(offspring)
        j = tree.leaves[i]
        while (j > 0) && (tree.offspring[j] == 0)
            push!(tree.free_indices, j)
            j = tree.parents[j]
            if j > 0
                tree.offspring[j] -= 1
            end
        end
    end
    return tree
end

function insert!(
    tree::ParticleTree{T}, states::Vector{T}, ancestors::AbstractVector{Int64}
) where {T}
    # parents of new generation
    parents = getindex(tree.leaves, ancestors)

    # ensure there are enough dead branches
    if (length(tree.free_indices) < length(ancestors))
        @debug "expanding tree"
        expand!(tree)
    end

    # find places for new states
    @inbounds for i in eachindex(states)
        tree.leaves[i] = pop!(tree.free_indices)
    end

    # insert new generation and update parent child relationships
    setindex!(tree.states, states, tree.leaves)
    setindex!(tree.parents, parents, tree.leaves)
    return tree
end

function expand!(tree::ParticleTree)
    M = length(tree)
    resize!(tree.states, 2 * M)

    # new allocations must be zero valued, this is not a perfect solution
    # tree.parents = [tree.parents; zero(tree.parents)]
    # tree.offspring = [tree.offspring; zero(tree.offspring)]
    append!(tree.parents, zeros(Int64, M))
    append!(tree.offspring, zeros(Int64, M))
    append_to_stack!(tree.free_indices, (2 * M):-1:(M + 1))
    return tree
end

function get_offspring(a::AbstractVector{Int64})
    offspring = zero(a)
    for i in a
        offspring[i] += 1
    end
    return offspring
end

function get_ancestry(tree::ParticleTree{T}) where {T}
    paths = Vector{Vector{T}}(undef, length(tree.leaves))
    @inbounds for (k, i) in enumerate(tree.leaves)
        j = tree.parents[i]
        xi = tree.states[i]

        xs = [xi]
        while j > 0
            push!(xs, tree.states[j])
            j = tree.parents[j]
        end
        paths[k] = reverse(xs)
    end
    return paths
end

function rand(rng::AbstractRNG, tree::ParticleTree, weights::AbstractVector{<:Real})
    b = StatsBase.sample(rng, StatsBase.Weights(weights))
    leaf = tree.leaves[b]

    j = tree.parents[leaf]
    xi = tree.states[leaf]

    xs = [xi]
    while j > 0
        push!(xs, tree.states[j])
        j = tree.parents[j]
    end
    return reverse(xs)
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
        getfield.(state.particles, :state),
        max(N, floor(Int64, N * log(N))),
    )
    return nothing
end

function (c::AncestorCallback)(
    model, filter, step, state, data, ::PostPredictCallback; kwargs...
)
    particles = state.particles
    prune!(c.tree, getfield.(particles, :ancestor))
    insert!(c.tree, getfield.(particles, :state), getfield.(particles, :ancestor))
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
