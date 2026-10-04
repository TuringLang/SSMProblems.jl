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

using AcceleratedKernels: searchsortedfirst
using CUDA
using Random: Random, AbstractRNG

# Explicit device destinations select the bundled GPU RNG. Allocating ordinary
# arrays and scalar draws continue to use the CPU child through the core interface.
function Random.rand!(rng::GeneralisedFilters.CombinedRNG, A::CUDA.AnyCuArray)
    return Random.rand!(GeneralisedFilters.gpu_rng(rng), A)
end
function Random.randn!(rng::GeneralisedFilters.CombinedRNG, A::CUDA.AnyCuArray)
    return Random.randn!(GeneralisedFilters.gpu_rng(rng), A)
end
# Resolve the intersection with GPUArrays' floating-array fallback.
function Random.randn!(
    rng::GeneralisedFilters.CombinedRNG,
    A::CUDA.AnyCuArray{<:Union{AbstractFloat,Complex{<:AbstractFloat}}},
)
    return Random.randn!(GeneralisedFilters.gpu_rng(rng), A)
end

GeneralisedFilters._host_array(x::CUDA.AnyCuArray) = Array(x)

## GPU RESAMPLING ##########################################################################

# Respect either a host RNG or a CUDA RNG; broadcasts below must use device arrays.
_device_uniforms(rng::AbstractRNG, ::Type{T}, n::Int) where {T} = CuArray(rand(rng, T, n))
_uniform_scalar(rng::AbstractRNG, ::Type{T}) where {T} = rand(rng, T)
function _device_uniforms(rng::GeneralisedFilters.CombinedRNG, ::Type{T}, n::Int) where {T}
    return Random.rand!(rng, CuArray{T}(undef, n))
end
# Systematic offsets and conditional reference offsets are host scalar decisions.
function _uniform_scalar(rng::GeneralisedFilters.CombinedRNG, ::Type{T}) where {T}
    return rand(GeneralisedFilters.cpu_rng(rng), T)
end
# CUDA 6 RNGs provide bulk draws but not Random's host scalar API.
function _uniform_scalar(rng::CUDA.RNG, ::Type{T}) where {T}
    return only(Array(_device_uniforms(rng, T, 1)))
end

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
    u0 = _uniform_scalar(rng, WT)
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
    uniform = _uniform_scalar(rng, WT)
    u, K = CUDA.@allowscalar GeneralisedFilters._reference_offset(uniform, r, ref_idx, n)
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

include("cuda/particle_tree.jl")

end # module CUDAExt
