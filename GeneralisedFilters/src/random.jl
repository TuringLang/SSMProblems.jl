export CombinedRNG, cpu_rng, gpu_rng

"""
    CombinedRNG(cpu::AbstractRNG, gpu::AbstractRNG)

Bundle explicitly owned CPU and GPU generators for the positional RNG argument.
Scalar draws, distributions and ordinary Julia arrays use `cpu`; supported GPU
array methods and GPU algorithms use `gpu`. Allocating `rand`/`randn` calls return
CPU arrays. Use [`cpu_rng`](@ref) and [`gpu_rng`](@ref) at backend boundaries.

Both children must be distinct, explicitly owned generators: nested bundles,
`TaskLocalRNG` and identical child objects are rejected. The bundle retains the
supplied generators, so drawing through either reference advances the same state.
`copy` copies both children; child generators must support `copy` for checkpoints.

`seed!(rng, seed::Integer)` reseeds both children. The CPU receives `seed`; the GPU
receives one `UInt64` drawn from a separately seeded copy of the CPU generator,
without consuming the live CPU stream. Children must support integer `seed!`.
Reproducibility requires preserving both states and the sequence of operations;
concurrent use requires the same care as using the child generators directly.
"""
struct CombinedRNG{C<:AbstractRNG,G<:AbstractRNG} <: AbstractRNG
    cpu::C
    gpu::G

    function CombinedRNG(cpu::C, gpu::G) where {C<:AbstractRNG,G<:AbstractRNG}
        (cpu isa CombinedRNG || gpu isa CombinedRNG) &&
            throw(ArgumentError("CombinedRNG children must not be nested bundles"))
        (cpu isa Random.TaskLocalRNG || gpu isa Random.TaskLocalRNG) &&
            throw(ArgumentError("CombinedRNG requires explicit RNGs; use Xoshiro() instead of TaskLocalRNG()"))
        cpu === gpu && throw(ArgumentError("CombinedRNG requires distinct CPU and GPU generators"))
        return new{C,G}(cpu, gpu)
    end
end

"""
    cpu_rng(rng::AbstractRNG)

Return the CPU child of a `CombinedRNG`, or `rng` itself for an ordinary RNG.
"""
cpu_rng(rng::AbstractRNG) = rng
cpu_rng(rng::CombinedRNG) = rng.cpu

"""
    gpu_rng(rng::AbstractRNG)

Return the GPU child of a `CombinedRNG`, or `rng` itself for an ordinary RNG.
This accessor selects a generator; it does not validate backend compatibility.
"""
gpu_rng(rng::AbstractRNG) = rng
gpu_rng(rng::CombinedRNG) = rng.gpu

Base.copy(rng::CombinedRNG) = CombinedRNG(copy(rng.cpu), copy(rng.gpu))

function Random.seed!(rng::CombinedRNG, seed::Integer)
    # Prepare and validate the CPU seed before changing either live generator.
    derivation_rng = copy(rng.cpu)
    Random.seed!(derivation_rng, seed)
    device_seed = rand(derivation_rng, UInt64)
    Random.seed!(rng.gpu, device_seed)
    Random.seed!(rng.cpu, seed)
    return rng
end

# Keep methods narrow: forwarding arbitrary rand arguments conflicts with the
# distribution and sampler methods provided by Random and Distributions.
const _CombinedRNGPrimitive = Union{
    Bool,Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64,Int128,UInt128,
    Float16,Float32,Float64,
}
Random.rand(rng::CombinedRNG) = rand(rng.cpu)
Random.rand(rng::CombinedRNG, range::AbstractRange) = rand(rng.cpu, range)
Random.rand(rng::CombinedRNG, ::Type{T}) where {T<:_CombinedRNGPrimitive} = rand(rng.cpu, T)
for T in (Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Int128, UInt128)
    @eval Random.rand(rng::CombinedRNG, sampler::Random.SamplerType{$T}) =
        rand(rng.cpu, sampler)
end

# Random's generic uniform/range samplers require this internal trait. Keep these
# compatibility hooks local and cover them with supported-version tests.
Random.rng_native_52(rng::CombinedRNG) = Random.rng_native_52(rng.cpu)
Random.rand(rng::CombinedRNG, sampler::Random.SamplerTrivial{Random.UInt52Raw{UInt64}}) =
    rand(rng.cpu, sampler)

Random.randn(rng::CombinedRNG) = randn(rng.cpu)
Random.randexp(rng::CombinedRNG) = Random.randexp(rng.cpu)
for f in (:randn, :randexp)
    @eval Random.$f(
        rng::CombinedRNG, T::Union{Type{Float16},Type{Float32},Type{Float64}}
    ) = Random.$f(rng.cpu, T)
end
Random.randn(rng::CombinedRNG, ::Type{Complex{T}}) where {T<:AbstractFloat} =
    randn(rng.cpu, Complex{T})

# Forward dense bulk paths to retain the CPU child's optimized implementations.
# Other CPU containers can use Random's generic elementwise sampler machinery.
Random.rand!(rng::CombinedRNG, A::Array) = Random.rand!(rng.cpu, A)
Random.rand!(rng::CombinedRNG, A::Array, ::Type{T}) where {T} =
    Random.rand!(rng.cpu, A, T)
Random.randn!(rng::CombinedRNG, A::Array{T}) where {T} = Random.randn!(rng.cpu, A)
Random.randexp!(rng::CombinedRNG, A::Array{T}) where {T} = Random.randexp!(rng.cpu, A)
