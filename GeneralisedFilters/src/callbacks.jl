"""Storage and callback methods for recording historic filtering information."""

using AcceleratedKernels
using DataStructures: Stack
using Random: rand

export AbstractCallback, CallbackTrigger
export PostInit, PostResample, PostPredict, PostUpdate
export PostInitCallback, PostResampleCallback, PostPredictCallback, PostUpdateCallback
export DenseParticleContainer, ParticleTree, ParallelParticleTree
export DenseAncestorCallback, AncestorCallback, ParallelAncestorCallback

## ABSTRACT CALLBACK SYSTEM ################################################################

abstract type AbstractCallback end

abstract type CallbackTrigger end

struct PostInitCallback <: CallbackTrigger end
const PostInit = PostInitCallback()
function (c::AbstractCallback)(model, filter, state, data, ::PostInitCallback; kwargs...)
    return nothing
end

struct PostResampleCallback <: CallbackTrigger end
const PostResample = PostResampleCallback()
function (c::AbstractCallback)(
    model, filter, step, state, data, ::PostResampleCallback; kwargs...
)
    return nothing
end

struct PostPredictCallback <: CallbackTrigger end
const PostPredict = PostPredictCallback()
function (c::AbstractCallback)(
    model, filter, step, state, data, ::PostPredictCallback; kwargs...
)
    return nothing
end

struct PostUpdateCallback <: CallbackTrigger end
const PostUpdate = PostUpdateCallback()
function (c::AbstractCallback)(
    model, filter, step, state, data, ::PostUpdateCallback; kwargs...
)
    return nothing
end

## DENSE PARTICLE STORAGE ##################################################################

struct DenseParticleContainer{T}
    particles::OffsetVector{Vector{T},Vector{Vector{T}}}
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
struct DenseAncestorCallback{T} <: AbstractCallback
    container::DenseParticleContainer{T}

    function DenseAncestorCallback(::Type{T}) where {T}
        particles = OffsetVector(Vector{T}[], -1)
        ancestors = Vector{Int}[]
        return new{T}(DenseParticleContainer(particles, ancestors))
    end
end

function (c::DenseAncestorCallback)(
    model, filter, state, data, ::PostInitCallback; kwargs...
)
    push!(c.container.particles, deepcopy(state.particles))
    return nothing
end

function (c::DenseAncestorCallback)(
    model, filter, step, state, data, ::PostUpdateCallback; kwargs...
)
    push!(c.container.particles, deepcopy(state.particles))
    push!(c.container.ancestors, deepcopy(state.ancestors))
    return nothing
end

## SPARSE PARTICLE STORAGE #################################################################

Base.append!(s::Stack, a::AbstractVector) = map(x -> push!(s, x), a)

"""
    ParticleTree

A sparse container for particle ancestry, which tracks the lineage of the filtered draws.

# Reference

Jacob, P., Murray L., & Rubenthaler S. (2015). Path storage in the particle 
filter [doi:10.1007/s11222-013-9445-x](https://dx.doi.org/10.1007/s11222-013-9445-x)
"""
mutable struct ParticleTree{T}
    states::Vector{T}
    parents::Vector{Int64}
    leaves::Vector{Int64}
    offspring::Vector{Int64}
    free_indices::Stack{Int64}

    function ParticleTree(states::Vector{T}, M::Integer) where {T}
        nodes = Vector{T}(undef, M)
        initial_free_indices = Stack{Int64}()
        append!(initial_free_indices, M:-1:(length(states) + 1))
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
    tree.parents = [tree.parents; zero(tree.parents)]
    tree.offspring = [tree.offspring; zero(tree.offspring)]
    append!(tree.free_indices, (2 * M):-1:(M + 1))
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
    b = randcat(rng, weights)
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

## GPU SPARSE PARTICLE STORAGE #############################################################

mutable struct ParallelParticleTree{ST,M<:CUDA.AbstractMemory}
    states::ST
    parents::CuVector{Int64,M}
    leaves::CuVector{Int64,M}
    offspring::CuVector{Int64,M}

    function ParallelParticleTree(states::ST, M::Integer) where {ST}
        if M < length(states)
            throw(ArgumentError("M must be greater than or equal to the number of states"))
        end

        parents = CUDA.zeros(Int64, M)
        offspring = CUDA.zeros(Int64, M)
        N = length(states)
        states = expand(states, M)
        tree_states = states
        leaves = CuArray(1:N)
        return new{ST,CUDA.DeviceMemory}(tree_states, parents, leaves, offspring)
    end
end

function scatter!(r, p, q)
    return r[q] .= p
end

function gather!(r, p, q)
    return r .= p[q]
end

function update_offspring!(offspring, leaves, parents)
    AcceleratedKernels.foreachindex(leaves) do i
        j = leaves[i]
        while (j > 0) && (offspring[j] == 0)
            j = parents[j]
            if j > 0
                offspring[j] -= 1
            end
        end
    end
end

function insert!(tree::ParallelParticleTree, states, ancestors::CuVector{Int64})
    b = CuVector{Int64}(undef, length(ancestors))
    gather!(b, tree.leaves, ancestors)

    # Update offspring counts
    offspring = ancestors_to_offspring(ancestors)
    scatter!(tree.offspring, offspring, tree.leaves)

    # Prune tree
    update_offspring!(tree.offspring, tree.leaves, tree.parents)

    # Expand tree if necessary
    if sum(tree.offspring .== 0) < length(ancestors)
        @debug "expanding tree"
        expand!(tree)
    end
    z = cumsum(tree.offspring .== 0)

    # Insert new states
    new_leaves = searchsortedfirst(z, CuArray(1:length(tree.leaves)))
    scatter!(tree.parents, b, new_leaves)
    tree.states[new_leaves] = states
    tree.leaves .= new_leaves
    return tree
end

function expand!(tree::ParallelParticleTree{T}) where {T}
    M = length(tree.states)

    new_parents = CUDA.zeros(Int64, 2M)
    new_parents[1:length(tree.parents)] = tree.parents
    tree.parents = new_parents

    new_offspring = CUDA.zeros(Int64, 2M)
    new_offspring[1:length(tree.offspring)] = tree.offspring
    tree.offspring = new_offspring

    tree.states = expand(tree.states, 2M)

    return tree
end

# Get ancestry of all particles
function get_ancestry(tree::ParallelParticleTree{ST}, T::Integer) where {ST}
    paths = OffsetVector(Vector{ST}(undef, T + 1), -1)
    parents = copy(tree.leaves)
    for t in T:-1:1
        paths[t] = tree.states[parents]
        gather!(parents, tree.parents, parents)
    end
    return paths
end

# Get ancestory of a single particle
function get_ancestry(
    container::ParallelParticleTree{ST}, i::Integer, T::Integer
) where {ST}
    path = OffsetVector(Vector{ST}(undef, T + 1), -1)
    CUDA.@allowscalar begin
        ancestor_index = container.leaves[i]
        for t in T:-1:0
            path[t] = container.states[ancestor_index]
            ancestor_index = container.parents[ancestor_index]
        end
        return path
    end
end

"""
    ParallelAncestorCallback

A callback for parallel sparse ancestry storage, which preallocates and returns a populated 
`ParallelParticleTree` object.
"""
# TODO: this should be initialised during inference so types don't need to be predetermined
struct ParallelAncestorCallback{T} <: AbstractCallback
    tree::ParallelParticleTree{T}
end

function (c::ParallelAncestorCallback)(
    model, filter, step, state, data, ::PostInitCallback; kwargs...
)
    @inbounds c.tree.states[1:(filter.N)] = deepcopy(state.particles)
    return nothing
end

function (c::ParallelAncestorCallback)(
    model, filter, step, state, data, ::PostUpdateCallback; kwargs...
)
    # insert! implicitly deepcopies
    insert!(c.tree, state.particles, state.ancestors)
    return nothing
end

## ANCESTOR STORAGE CALLBACK ###############################################################

"""
    AncestorCallback

A callback for sparse ancestry storage, which preallocates and returns a populated 
`ParticleTree` object.
"""
struct AncestorCallback{T} <: AbstractCallback
    tree::ParticleTree{T}
end

function AncestorCallback(::Type{T}, N::Integer, C::Real=1.0) where {T}
    M = floor(Int64, C * N * log(N))
    nodes = Vector{T}(undef, N)
    return new{T}(ParticleTree(nodes, M))
end

function (c::AncestorCallback)(model, filter, state, data, ::PostInitCallback; kwargs...)
    @inbounds c.tree.states[1:(filter.N)] = deepcopy(state.particles)
    return nothing
end

function (c::AncestorCallback)(
    model, filter, step, state, data, ::PostPredictCallback; kwargs...
)
    prune!(c.tree, get_offspring(state.ancestors))
    insert!(c.tree, state.particles, state.ancestors)
    return nothing
end

"""
    ResamplerCallback

A callback which follows the resampling indices over the filtering algorithm. This is more
of a debug tool and visualizer for various resapmling algorithms.
"""
struct ResamplerCallback <: AbstractCallback
    tree::ParticleTree

    function ResamplerCallback(N::Integer, C::Real=1.0)
        M = floor(Int64, C * N * log(N))
        nodes = collect(1:N)
        return new(ParticleTree(nodes, M))
    end
end

function (c::ResamplerCallback)(
    model, filter, step, state, data, ::PostResampleCallback; kwargs...
)
    if step != 1
        prune!(c.tree, get_offspring(state.ancestors))
        insert!(c.tree, collect(1:(filter.N)), state.ancestors)
    end
    return nothing
end
