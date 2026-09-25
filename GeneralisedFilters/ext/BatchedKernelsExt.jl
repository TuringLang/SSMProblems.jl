module BatchedKernelsExt

using GeneralisedFilters: GeneralisedFilters, Particle, ParticleDistribution
using BatchedKernels:
    BatchedStruct,
    BatchedCuScalar,
    BatchedCuVector,
    BatchedCuMatrix,
    SharedCuVector,
    SharedCuMatrix
using CUDA

include("batched_kernels/containers.jl")
include("batched_kernels/models.jl")

end
