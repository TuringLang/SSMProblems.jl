module BatchedKernelsExt

using GeneralisedFilters: GeneralisedFilters, Particle, ParticleDistribution
using BatchedKernels:
    BatchedStruct,
    BatchedRNG,
    fuse,
    shared,
    BatchedCuScalar,
    BatchedCuVector,
    BatchedCuMatrix,
    SharedCuVector,
    SharedCuMatrix
using BatchedKernels: SharedValue, allocate_batch, check_batch_copy
using CUDA
using Random: AbstractRNG

include("batched_kernels/containers.jl")
include("batched_kernels/particle_tree.jl")
include("batched_kernels/models.jl")
include("batched_kernels/filtering.jl")
include("batched_kernels/backward.jl")

end
