"""
CUDA extension for GeneralisedFilters.

This extension provides GPU-accelerated particle filtering operations:
- GPU resampling methods (Multinomial, Systematic, Stratified) for CuVector weights
- Offspring/ancestor conversion kernels
- ParallelParticleTree for GPU-based sparse particle storage
- ParallelAncestorCallback for GPU ancestry tracking
"""
module CUDAExt

using GeneralisedFilters:
    GeneralisedFilters,
    Multinomial,
    Systematic,
    Stratified,
    AbstractCallback,
    PostInitCallback,
    PostUpdateCallback,
    num_particles

using GeneralisedFilters.OffsetArrays: OffsetVector

using AcceleratedKernels: searchsortedfirst, foreachindex
using CUDA
import CUDACore: AbstractMemory, DeviceMemory
using Random: AbstractRNG

## GPU RESAMPLING ##########################################################################

# Following Code 5 of Murray et. al (2015)
function GeneralisedFilters.sample_ancestors(
    rng::AbstractRNG, ::Multinomial, weights::CuVector{WT}, n::Int=length(weights)
) where {WT}
    W = cumsum(weights)
    Wn = CUDA.@allowscalar W[n]
    us = CUDA.rand(WT, n) * Wn
    as = searchsortedfirst(W, us)
    return as
end

function GeneralisedFilters.sample_ancestors(
    rng::AbstractRNG, ::Systematic, weights::CuVector, n::Int=length(weights)
)
    offspring = sample_offspring(rng, Systematic(), weights, n)
    return offspring_to_ancestors(offspring)
end

# Following Code 8 of Murray et. al (2015)
function sample_offspring(
    rng::AbstractRNG, ::Systematic, weights::CuVector{WT}, n::Int=length(weights)
) where {WT}
    W = cumsum(weights)
    Wn = CUDA.@allowscalar W[n]
    u0 = CUDA.@allowscalar rand(rng, WT)
    r = n * W / Wn
    offspring = min.(n, floor.(Int, r .+ u0))
    return offspring
end

function GeneralisedFilters.sample_ancestors(
    rng::AbstractRNG, ::Stratified, weights::CuVector, n::Int=length(weights)
)
    offspring = sample_offspring(rng, Stratified(), weights, n)
    return offspring_to_ancestors(offspring)
end

# Following Code 7 of Murray et. al (2015)
function sample_offspring(
    rng::AbstractRNG, ::Stratified, weights::CuVector{WT}, n::Int=length(weights)
) where {WT}
    u = rand(rng, n)
    W = cumsum(weights)
    Wn = CUDA.@allowscalar W[n]
    r = n * W / Wn
    k = min.(n, floor.(Int, r .+ 1))
    offspring = min.(n, floor.(Int, r .+ u[k]))
    return offspring
end

## GPU SCALAR INDEXING FOR REF_STATE #######################################################

# Override _set_ref_index! for CuVector to use @allowscalar
function GeneralisedFilters._set_ref_index!(idxs::CuVector)
    CUDA.@allowscalar idxs[1] = 1
    return nothing
end

## ANCESTOR-OFFSPRING CONVERSION ###########################################################

function _offspring_to_ancestors_kernel!(ancestors, offspring, N)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    @inbounds for i in index:stride:N
        start = i == 1 ? 0 : offspring[i - 1]
        finish = offspring[i]
        for j in (start + 1):finish
            ancestors[j] = i
        end
    end

    return nothing
end

function offspring_to_ancestors(offspring::CuVector{<:Integer})
    N = length(offspring)
    ancestors = similar(offspring)

    threads = 256
    blocks = ceil(Int, N / threads)

    @cuda threads = threads blocks = blocks _offspring_to_ancestors_kernel!(
        ancestors, offspring, N
    )

    return ancestors
end

function _ancestors_to_offspring_kernel!(output, ancestors, N)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    @inbounds for i in index:stride:N
        CUDA.@atomic output[ancestors[i]] += 1
    end

    return nothing
end

function ancestors_to_offspring(ancestors::CuVector{Int})
    N = length(ancestors)
    offspring = CUDA.zeros(Int, N)

    threads = 256
    blocks = ceil(Int, N / threads)

    @cuda threads = threads blocks = blocks _ancestors_to_offspring_kernel!(
        offspring, ancestors, N
    )

    return offspring
end

## GPU SPARSE PARTICLE STORAGE #############################################################

mutable struct ParallelParticleTree{ST,M<:AbstractMemory}
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
        return new{ST,DeviceMemory}(tree_states, parents, leaves, offspring)
    end
end

function scatter!(r, p, q)
    return r[q] .= p
end

function gather!(r, p, q)
    return r .= p[q]
end

function update_offspring!(offspring, leaves, parents)
    foreachindex(leaves) do i
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

# Get ancestry of a single particle
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

# Helper for expanding state arrays (placeholder - actual implementation may vary)
function expand(states, M)
    # This needs to match the actual state type
    new_states = similar(states, M)
    new_states[1:length(states)] = states
    return new_states
end

## GPU ANCESTOR CALLBACK ###################################################################

"""
    ParallelAncestorCallback

A callback for parallel sparse ancestry storage, which preallocates and returns a populated
`ParallelParticleTree` object.
"""
struct ParallelAncestorCallback{T} <: AbstractCallback
    tree::ParallelParticleTree{T}
end

function (c::ParallelAncestorCallback)(
    model, filter, step, state, data, ::PostInitCallback; kwargs...
)
    N = num_particles(filter)
    @inbounds c.tree.states[1:N] = deepcopy(state.particles)
    return nothing
end

function (c::ParallelAncestorCallback)(
    model, filter, step, state, data, ::PostUpdateCallback; kwargs...
)
    # insert! implicitly deepcopies
    particles = state.particles
    insert!(c.tree, getfield.(particles, :state), getfield.(particles, :ancestor))
    return nothing
end

end # module CUDAExt
