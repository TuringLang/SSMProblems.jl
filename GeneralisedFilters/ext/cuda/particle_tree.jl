using GeneralisedFilters: _ParallelParticleTree
import Base: insert!

# Retain the old extension-local type name and three-argument extraction method.
const ParallelParticleTree = _ParallelParticleTree

GeneralisedFilters._tree_device(x::CUDA.AnyCuVector) = CUDA.device(x) == CUDA.device()
GeneralisedFilters._tree_allocate(x::CUDA.AnyCuVector, n) = similar(x, n)
function GeneralisedFilters._tree_check(dest::CUDA.AnyCuVector, src::CUDA.AnyCuVector)
    eltype(dest) === eltype(src) ||
        throw(ArgumentError("particle state scalar types differ"))
    CUDA.device(dest) == CUDA.device(src) ||
        throw(ArgumentError("particle states must reside on the same device"))
    Base.mightalias(dest, src) &&
        throw(ArgumentError("tree input must not alias tree storage"))
    return nothing
end
function GeneralisedFilters._tree_scatter!(
    dest::CUDA.AnyCuVector, idxs, src::CUDA.AnyCuVector
)
    return setindex!(dest, src, idxs)
end
# Scalar payloads are transferred only after gathering the selected path. Vector
# and composite payloads have their own compact, device-backed representation.
GeneralisedFilters._tree_members(x::CUDA.AnyCuVector) = Array(x)

function GeneralisedFilters._parallel_tree(initial, prototype, capacity::Integer)
    isempty(initial) &&
        throw(ArgumentError("particle history requires a nonempty initial population"))
    capacity > 0 || throw(ArgumentError("particle tree capacity must be positive"))
    GeneralisedFilters._tree_device(initial) &&
    GeneralisedFilters._tree_device(prototype) || throw(
        ArgumentError(
            "GPU particle history requires numerical leaves on the active CUDA device"
        ),
    )
    # Allocation can change view and composite types. Parameterise by the owned
    # storage, not by the input's possibly shared or view-backed representation.
    initial_copy = GeneralisedFilters._tree_allocate(initial, length(initial))
    GeneralisedFilters._tree_scatter!(initial_copy, Base.OneTo(length(initial)), initial)
    states = GeneralisedFilters._tree_allocate(prototype, capacity)
    parents = CUDA.zeros(Int64, capacity)
    offspring = CUDA.zeros(Int64, capacity)
    leaves = CuArray(collect(Int64, 1:length(initial)))
    return _ParallelParticleTree(initial_copy, states, parents, leaves, offspring, 0)
end

function GeneralisedFilters._ParallelParticleTree(initial, capacity::Integer)
    return GeneralisedFilters._parallel_tree(initial, initial, capacity)
end
function GeneralisedFilters._ParallelParticleTree(
    initial, states, ancestors, capacity::Integer
)
    tree = GeneralisedFilters._parallel_tree(initial, states, capacity)
    return insert!(tree, states, ancestors)
end

function GeneralisedFilters.ParticleTree(initial::CUDA.AnyCuVector, capacity::Integer)
    return ParallelParticleTree(initial, capacity)
end
function GeneralisedFilters.ParticleTree(
    initial::CUDA.AnyCuVector, states::CUDA.AnyCuVector, ancestors, capacity::Integer
)
    return ParallelParticleTree(initial, states, ancestors, capacity)
end

function _prune_kernel!(offspring, leaves, parents, counts, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n && counts[i] == 0
        # counts is immutable for the duration of pruning. Every dead old leaf
        # starts exactly one walk; old leaves cannot be ancestors of each other.
        j = parents[leaves[i]]
        while j > 0
            old = CUDA.atomic_sub!(pointer(offspring, j), Int64(1))
            old == 1 || break
            # Only the thread owning the 1 -> 0 transition propagates upward.
            # Parent pointers stay immutable until this kernel completes.
            j = parents[j]
        end
    end
    return nothing
end

function insert!(tree::ParallelParticleTree, states, ancestors::CUDA.AnyCuVector{<:Integer})
    GeneralisedFilters._validate_history_step(states, ancestors, length(tree.leaves))
    eltype(ancestors) === Bool && throw(ArgumentError("Boolean ancestry is unsupported"))
    GeneralisedFilters._tree_check(tree.states, states)
    CUDA.device(ancestors) == CUDA.device(tree.leaves) ||
        throw(ArgumentError("ancestry and tree must reside on the same device"))
    n = length(tree.leaves)
    # Ancestors index the preceding population, never pool slots. Keep metadata
    # Int64 independently of the incoming Int32/Int64 ancestry representation.
    a = Int64.(ancestors)
    parents_of_new = tree.depth == 0 ? -tree.leaves[a] : tree.leaves[a]
    if tree.depth > 0
        counts = ancestors_to_offspring(a)
        tree.offspring[tree.leaves] = counts
        @cuda threads=256 blocks=cld(n, 256) _prune_kernel!(
            tree.offspring, tree.leaves, tree.parents, counts, n
        )
    end
    # Stream-ordered kernels and reductions separate pruning from reclamation.
    # A zero count denotes a free slot at this phase (no current leaves yet).
    free_count = count(iszero, tree.offspring)
    while free_count < n
        old_capacity = length(tree.states)
        expand!(tree)
        free_count += old_capacity
    end
    ranks = cumsum(tree.offspring .== 0)
    new_leaves = Int64.(searchsortedfirst(ranks, CuArray(collect(Int64, 1:n))))
    GeneralisedFilters._tree_scatter!(tree.states, new_leaves, states)
    tree.parents[new_leaves] = parents_of_new
    # Current leaves have a temporary reference of one. On the next insertion
    # this is replaced by their actual offspring count before pruning starts.
    tree.offspring[new_leaves] .= 1
    tree.leaves = new_leaves
    tree.depth += 1
    return tree
end

function expand!(tree::ParallelParticleTree)
    capacity = length(tree.states)
    capacity <= typemax(Int) ÷ 2 || throw(OverflowError("particle tree capacity overflow"))
    live = findall(!iszero, tree.offspring)
    states = GeneralisedFilters._tree_allocate(tree.states, 2capacity)
    # Never read payload slots that have not been initialised.
    GeneralisedFilters._tree_scatter!(states, live, tree.states[live])
    parents = CUDA.zeros(Int64, 2capacity)
    offspring = CUDA.zeros(Int64, 2capacity)
    parents[1:capacity] = tree.parents
    offspring[1:capacity] = tree.offspring
    tree.states, tree.parents, tree.offspring = states, parents, offspring
    return tree
end

function Base.push!(
    tree::ParallelParticleTree, state::GeneralisedFilters.ParticleDistribution
)
    return insert!(
        tree,
        GeneralisedFilters._history_states(state),
        GeneralisedFilters._history_ancestors(state),
    )
end

function GeneralisedFilters.get_ancestry(tree::ParallelParticleTree, i::Integer)
    checkbounds(tree.leaves, i)
    indices = Vector{Int64}(undef, tree.depth)
    # Intentional scalar metadata transfers. Payloads are gathered in bulk once
    # the path is known, so no full population reaches the host or escapes by view.
    j = CUDA.@allowscalar tree.leaves[i]
    for t in tree.depth:-1:1
        indices[t] = j
        j = CUDA.@allowscalar tree.parents[j]
    end
    root = tree.depth == 0 ? j : -j
    initial = GeneralisedFilters._tree_members(tree.initial_states[[root]])
    xs = GeneralisedFilters._tree_members(tree.states[CuArray(indices)])
    return ReferenceTrajectory(only(initial), xs)
end

function GeneralisedFilters.get_ancestry(tree::ParallelParticleTree)
    return [GeneralisedFilters.get_ancestry(tree, i) for i in eachindex(tree.leaves)]
end

# The former extension API took a supplied depth. Keep it checked, while the
# public GF API uses the depth owned by the container.
function _check_tree_depth(tree, depth)
    return depth == tree.depth ||
           throw(ArgumentError("trajectory depth differs from tree history"))
end
function get_ancestry(tree::ParallelParticleTree, depth::Integer)
    _check_tree_depth(tree, depth)
    return GeneralisedFilters.get_ancestry(tree)
end
function get_ancestry(tree::ParallelParticleTree, i::Integer, depth::Integer)
    _check_tree_depth(tree, depth)
    return GeneralisedFilters.get_ancestry(tree, i)
end

function GeneralisedFilters._update_tree!(
    tree::ParallelParticleTree, state::GeneralisedFilters.ParticleDistribution
)
    return push!(tree, state)
end
function GeneralisedFilters._sample_trajectory(
    rng::AbstractRNG,
    tree::ParallelParticleTree,
    state::GeneralisedFilters.ParticleDistribution,
)
    return GeneralisedFilters.get_ancestry(
        tree, GeneralisedFilters._sample_index(rng, GeneralisedFilters.get_weights(state))
    )
end
