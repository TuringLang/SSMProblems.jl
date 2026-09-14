"""
CUDA extension for GeneralisedFilters.

This extension provides GPU-accelerated particle filtering operations:
- GPU resampling methods (Multinomial, Systematic, Stratified) for CuVector weights,
  including their conditional variants for conditional SMC
- Offspring/ancestor conversion kernels
- ParallelParticleTree for GPU-based sparse particle storage
"""
module CUDAExt

using GeneralisedFilters: GeneralisedFilters, Multinomial, Systematic, Stratified

using GeneralisedFilters: ReferenceTrajectory

using AcceleratedKernels: searchsortedfirst, foreachindex
using CUDA
using Random: AbstractRNG

## GPU RESAMPLING ##########################################################################

# Respect either a host RNG or a CUDA RNG; broadcasts below must use device arrays.
_device_uniforms(rng::AbstractRNG, ::Type{T}, n::Int) where {T} = CuArray(rand(rng, T, n))

function _validate_resampling_count(weights, n)
    n >= 0 || throw(ArgumentError("sample count must be nonnegative"))
    n > 0 && isempty(weights) && throw(ArgumentError("cannot sample from empty weights"))
    return nothing
end

# Following Code 5 of Murray et. al (2015)
function GeneralisedFilters.sample_ancestors(
    rng::AbstractRNG, ::Multinomial, weights::CuVector{WT}, n::Int=length(weights)
) where {WT}
    _validate_resampling_count(weights, n)
    n == 0 && return CUDA.zeros(Int, 0)
    W = cumsum(weights)
    Wn = CUDA.@allowscalar W[end]
    us = _device_uniforms(rng, WT, n) .* Wn
    as = searchsortedfirst(W, us)
    return as
end

# Multinomial resampling is exchangeable, so the reference may be placed in a fixed slot
# with the remaining slots drawn by the ordinary rule.
function GeneralisedFilters.conditional_sample_ancestors(
    rng::AbstractRNG, ::Multinomial, weights::CuVector, ref_idx::Integer
)
    as = GeneralisedFilters.sample_ancestors(rng, Multinomial(), weights)
    CUDA.@allowscalar as[1] = ref_idx
    return as
end

function GeneralisedFilters.sample_ancestors(
    rng::AbstractRNG, ::Systematic, weights::CuVector, n::Int=length(weights)
)
    offspring = sample_offspring(rng, Systematic(), weights, n)
    return offspring_to_ancestors(offspring)
end

function GeneralisedFilters.conditional_sample_ancestors(
    rng::AbstractRNG, ::Systematic, weights::CuVector, ref_idx::Integer
)
    offspring, K = sample_conditional_offspring(rng, Systematic(), weights, ref_idx)
    return offspring_to_ancestors(offspring; shift=K - 1)
end

# Following Code 8 of Murray et. al (2015)
function sample_offspring(
    rng::AbstractRNG, ::Systematic, weights::CuVector{WT}, n::Int=length(weights)
) where {WT}
    _validate_resampling_count(weights, n)
    n == 0 && return CUDA.zeros(Int, length(weights))
    W = cumsum(weights)
    Wn = CUDA.@allowscalar W[end]
    u0 = CUDA.@allowscalar rand(rng, WT)
    r = n * W / Wn
    offspring = min.(n, floor.(Int, r .+ u0))
    return offspring
end

"""
    _conditional_offset(rng, r, ref_idx, n) -> (u, K)

Offset and reference slot for conditional systematic/stratified resampling, in the offset
convention used by the offspring codes of Murray et al. (2015).

Those codes compute `floor(r + u)`, which retains slot `m` for a particle whenever
`u >= 1 - frac(r)`, whereas the stratified inverse-CDF form retains it when the offset is
`<= frac(r)`. The two agree in distribution for a uniform offset, so the unconditional
codes are unaffected, but the conditional offset must be complemented to land in the
reference particle's own interval.
"""
function _conditional_offset(
    rng::AbstractRNG, r::CuVector{WT}, ref_idx::Integer, n::Int
) where {WT}
    u, K = CUDA.@allowscalar GeneralisedFilters._reference_offset(rng, r, ref_idx, n)
    return one(WT) - u, K
end

function sample_conditional_offspring(
    rng::AbstractRNG, ::Systematic, weights::CuVector{WT}, ref_idx::Integer
) where {WT}
    n = length(weights)
    _validate_resampling_count(weights, n)
    W = cumsum(weights)
    Wn = CUDA.@allowscalar W[end]
    r = n * W / Wn
    u0, K = _conditional_offset(rng, r, ref_idx, n)
    offspring = min.(n, floor.(Int, r .+ u0))
    return offspring, K
end

function GeneralisedFilters.sample_ancestors(
    rng::AbstractRNG, ::Stratified, weights::CuVector, n::Int=length(weights)
)
    offspring = sample_offspring(rng, Stratified(), weights, n)
    return offspring_to_ancestors(offspring)
end

function GeneralisedFilters.conditional_sample_ancestors(
    rng::AbstractRNG, ::Stratified, weights::CuVector, ref_idx::Integer
)
    offspring, K = sample_conditional_offspring(rng, Stratified(), weights, ref_idx)
    return offspring_to_ancestors(offspring; shift=K - 1)
end

# Following Code 7 of Murray et. al (2015)
function sample_offspring(
    rng::AbstractRNG, ::Stratified, weights::CuVector{WT}, n::Int=length(weights)
) where {WT}
    _validate_resampling_count(weights, n)
    n == 0 && return CUDA.zeros(Int, length(weights))
    u = _device_uniforms(rng, WT, n)
    W = cumsum(weights)
    Wn = CUDA.@allowscalar W[end]
    r = n * W / Wn
    k = min.(n, floor.(Int, r .+ 1))
    offspring = min.(n, floor.(Int, r .+ u[k]))
    return offspring
end

function sample_conditional_offspring(
    rng::AbstractRNG, ::Stratified, weights::CuVector{WT}, ref_idx::Integer
) where {WT}
    n = length(weights)
    _validate_resampling_count(weights, n)
    u = _device_uniforms(rng, WT, n)
    W = cumsum(weights)
    Wn = CUDA.@allowscalar W[end]
    r = n * W / Wn
    # Every slot but the reference's keeps its own independent offset.
    u_ref, K = _conditional_offset(rng, r, ref_idx, n)
    CUDA.@allowscalar u[K] = u_ref
    k = min.(n, floor.(Int, r .+ 1))
    offspring = min.(n, floor.(Int, r .+ u[k]))
    return offspring, K
end

## ANCESTOR-OFFSPRING CONVERSION ###########################################################

function _offspring_to_ancestors_kernel!(ancestors, offspring, N, total, shift)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    @inbounds for i in index:stride:N
        start = i == 1 ? 0 : offspring[i - 1]
        finish = offspring[i]
        for j in (start + 1):finish
            # Cyclical shift, which brings the reference slot to the front for conditional
            # resampling. `shift < total`, so one wrap-around is enough.
            slot = j - shift
            ancestors[slot < 1 ? slot + total : slot] = i
        end
    end

    return nothing
end

"""
    offspring_to_ancestors(offspring; shift=0)

Expand a cumulative offspring vector into ancestor indices, cyclically shifted so that slot
`shift + 1` of the scheme's own ordering comes first.
"""
function offspring_to_ancestors(offspring::CuVector{<:Integer}; shift::Integer=0)
    N = length(offspring)
    total = N == 0 ? 0 : CUDA.@allowscalar offspring[end]
    ancestors = similar(offspring, total)
    total == 0 && return ancestors

    threads = 256
    blocks = ceil(Int, N / threads)

    @cuda threads = threads blocks = blocks _offspring_to_ancestors_kernel!(
        ancestors, offspring, N, total, shift
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
    N == 0 && return offspring

    threads = 256
    blocks = ceil(Int, N / threads)

    @cuda threads = threads blocks = blocks _ancestors_to_offspring_kernel!(
        offspring, ancestors, N
    )

    return offspring
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

# Get ancestry of all particles. The parallel tree stores initial and subsequent states
# in a single `states` buffer (homogeneous type), so the returned trajectories have
# T0 == T.
function get_ancestry(tree::ParallelParticleTree{ST}, T::Integer) where {ST}
    buf = Vector{Vector{eltype(tree.states)}}(undef, T + 1)
    parents = copy(tree.leaves)
    for t in (T + 1):-1:2
        buf[t] = Vector(tree.states[parents])
        gather!(parents, tree.parents, parents)
    end
    buf[1] = Vector(tree.states[parents])
    # Each leaf's trajectory: x0 = buf[1][k], xs = [buf[2][k], ..., buf[T+1][k]]
    return [
        ReferenceTrajectory(buf[1][k], [buf[t + 1][k] for t in 1:T]) for
        k in eachindex(tree.leaves)
    ]
end

# Get ancestry of a single particle
function get_ancestry(
    container::ParallelParticleTree{ST}, i::Integer, T::Integer
) where {ST}
    xs = Vector{eltype(container.states)}(undef, T)
    CUDA.@allowscalar begin
        ancestor_index = container.leaves[i]
        for t in T:-1:1
            xs[t] = container.states[ancestor_index]
            ancestor_index = container.parents[ancestor_index]
        end
        x0 = container.states[ancestor_index]
        return ReferenceTrajectory(x0, xs)
    end
end

# Helper for expanding state arrays (placeholder - actual implementation may vary)
function expand(states, M)
    # This needs to match the actual state type
    new_states = similar(states, M)
    new_states[1:length(states)] = states
    return new_states
end

end # module CUDAExt
