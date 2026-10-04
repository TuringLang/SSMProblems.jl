const TreeBatch = Union{
    BatchedCuScalar,
    BatchedCuVector,
    BatchedCuMatrix,
    BatchedStruct,
    SharedCuVector,
    SharedCuMatrix,
    SharedScalar,
    SharedValue,
}

function GeneralisedFilters._tree_device(
    x::Union{BatchedCuScalar,BatchedCuVector,BatchedCuMatrix,SharedCuVector,SharedCuMatrix}
)
    return x.data isa CUDA.AnyCuArray && CUDA.device(x.data) == CUDA.device()
end
GeneralisedFilters._tree_device(x::Union{SharedScalar,SharedValue}) = true
function GeneralisedFilters._tree_device(x::BatchedStruct)
    return all(GeneralisedFilters._tree_device, values(x.components))
end
GeneralisedFilters._tree_allocate(x::TreeBatch, n) = allocate_batch(x, n)
function GeneralisedFilters._tree_check(dest::TreeBatch, src::TreeBatch)
    return check_batch_copy(dest, src)
end
function GeneralisedFilters._tree_scatter!(dest::TreeBatch, indices, src::TreeBatch)
    return setindex!(dest, src, indices)
end

# The batch has already been gathered into compact independent storage. Matrix
# and vector members remain device views of that small allocation. Only selected
# scalar fields are transferred to reconstruct Julia structs (never populations).
function GeneralisedFilters._tree_members(x::TreeBatch)
    return CUDA.@allowscalar [x[i] for i in 1:length(x)]
end

function GeneralisedFilters.ParticleTree(initial::TreeBatch, capacity::Integer)
    return GeneralisedFilters._parallel_tree(initial, initial, capacity)
end
function GeneralisedFilters.ParticleTree(
    initial::TreeBatch, states::AbstractVector, ancestors, capacity::Integer
)
    tree = GeneralisedFilters._parallel_tree(initial, states, capacity)
    return insert!(tree, states, ancestors)
end
function GeneralisedFilters.ParticleTree(
    initial::CUDA.AnyCuVector, states::TreeBatch, ancestors, capacity::Integer
)
    tree = GeneralisedFilters._parallel_tree(initial, states, capacity)
    return insert!(tree, states, ancestors)
end

_outer_batch(states) = states
_outer_batch(states::BatchedStruct{<:GeneralisedFilters.RBState}) = states.components.x
function GeneralisedFilters._outer_history_states(
    state::ParticleDistribution{W,P,B}
) where {W,P<:Particle,B<:BatchedStruct{P}}
    return _outer_batch(GeneralisedFilters._history_states(state))
end
