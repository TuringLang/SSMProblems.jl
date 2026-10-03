module BatchedKernelsExt

using GeneralisedFilters: GeneralisedFilters, Particle, ParticleDistribution
using BatchedKernels:
    BatchedStruct,
    BatchedRNG,
    fuse,
    BatchedCuScalar,
    BatchedCuVector,
    BatchedCuMatrix,
    SharedCuVector,
    SharedCuMatrix
using CUDA
using Random: AbstractRNG

include("batched_kernels/containers.jl")
include("batched_kernels/models.jl")
include("batched_kernels/filtering.jl")
include("batched_kernels/backward.jl")

end
