# Representation-independent declarations. CUDA owns genealogy operations; the
# CUDA and BatchedKernels extensions implement payload operations independently.
mutable struct _ParallelParticleTree{I,S,A}
    initial_states::I
    states::S
    parents::A              # negative values index initial_states
    leaves::A
    offspring::A
    depth::Int
end

function _parallel_tree end
function _tree_allocate end
function _tree_check end
function _tree_scatter! end
function _tree_members end
_tree_device(::Any) = false

# Only the conditional trajectory recorder projects states. Public histories
# retain full states, including the beliefs accompanying surviving particles.
struct _OuterStateHistory{H}
    history::H
end
function _outer_history_states end
