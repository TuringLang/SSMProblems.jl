export AbstractExecution, SerialExecution, ThreadedExecution, GPUExecution

"""
    AbstractExecution

How a particle filter evaluates its particle population. Choose it with the `execution`
keyword of [`ParticleFilter`](@ref). `RBPF`, `AuxiliaryParticleFilter` and
`ConditionalSMC` use the execution of the particle filter they contain.
"""
abstract type AbstractExecution end

"""
    SerialExecution()

Evaluate particles in order on the calling task, drawing directly from the filter's random
number generator. This is the default.
"""
struct SerialExecution <: AbstractExecution end

"""
    GPUExecution()

Initialise a device particle population using the model's GPU initialisation method.
Subsequent operations dispatch on that population's storage, while time iteration and
scalar decisions remain host-controlled. This setting does not automatically upload
an arbitrary CPU model or make its callbacks GPU-compatible.

Model implementations can extend `initialise(::GPUExecution, rng, prior, algo; ref_state)`
to select their resident device parameters and construct batched particle storage.
The BatchedKernels extension provides this method for supported device Gaussian priors.
"""
struct GPUExecution <: AbstractExecution end

"""
    ThreadedExecution(; blocksize=32, ntasks=nothing)

Evaluate particles concurrently on up to `ntasks` tasks (`Threads.nthreads()` when
`nothing`).

Particles are evaluated in consecutive blocks of `blocksize`. Each population draw takes
one value from the filter's random number generator and derives an independent generator
for every block from it. Results therefore depend on the generator state, the number of
particles and `blocksize`, but not on `ntasks`, the number of threads or scheduling. They
differ from [`SerialExecution`](@ref), which draws from the filter's generator directly.
Deterministic particle computations, such as observation updates and ancestor weights,
match serial evaluation exactly. Resampling and weight normalisation remain serial.

`blocksize` is also the smallest unit of work given to a task: prefer larger blocks when
each particle is cheap and smaller blocks for expensive particles, such as
Rao–Blackwellised ones.

Model code must be thread-safe: draw randomness only from the generator it is given and do
not mutate shared state. Every particle must have the same concrete type. Reverse-mode
automatic differentiation through threaded evaluation is not supported.
"""
struct ThreadedExecution <: AbstractExecution
    blocksize::Int
    ntasks::Union{Nothing,Int}
    function ThreadedExecution(blocksize::Integer, ntasks::Union{Nothing,Integer})
        blocksize > 0 || throw(ArgumentError("blocksize must be positive"))
        isnothing(ntasks) || ntasks > 0 || throw(ArgumentError("ntasks must be positive"))
        return new(blocksize, ntasks)
    end
end

function ThreadedExecution(; blocksize::Integer=32, ntasks::Union{Nothing,Integer}=nothing)
    return ThreadedExecution(blocksize, ntasks)
end

## POPULATION MAPS #########################################################################

# Every per-particle traversal goes through `_population_map`. Maps that consume randomness
# receive the generator to use for each particle as the first argument of `f`.
_population_map(f, ::SerialExecution, n::Integer) = map(f, 1:n)
function _population_map(f, ::SerialExecution, rng::AbstractRNG, n::Integer)
    return map(i -> f(rng, i), 1:n)
end

function _population_map(f, ex::ThreadedExecution, n::Integer)
    return _blocked_map((_, i) -> f(i), Returns(nothing), ex, n)
end
function _population_map(f, ex::ThreadedExecution, rng::AbstractRNG, n::Integer)
    key = rand(rng, UInt64)
    return _blocked_map(f, block -> _block_rng(key, block), ex, n)
end

# Workers claim whole blocks in any order. Everything a block computes depends only on the
# block index, so the result does not depend on how blocks are scheduled.
function _blocked_map(f, block_context, ex::ThreadedExecution, n::Integer)
    bs = ex.blocksize
    nblocks = cld(n, bs)
    context = block_context(1)
    x = f(context, 1)
    out = Vector{typeof(x)}(undef, n)
    out[1] = x

    next_block = Threads.Atomic{Int}(2)
    function run_blocks!()
        while (block = Threads.atomic_add!(next_block, 1)) <= nblocks
            block_ctx = block_context(block)
            for i in ((block - 1) * bs + 1):min(block * bs, n)
                _store!(out, i, f(block_ctx, i))
            end
        end
        return nothing
    end

    ntasks = min(something(ex.ntasks, Threads.nthreads()), nblocks)
    tasks = [Threads.@spawn(run_blocks!()) for _ in 2:ntasks]
    try
        for i in 2:min(bs, n)
            _store!(out, i, f(context, i))
        end
        run_blocks!()
    finally
        foreach(_wait_task, tasks)
    end
    return out
end

_store!(out::Vector{T}, i::Integer, x::T) where {T} = (out[i]=x; nothing)
function _store!(out::Vector{T}, i::Integer, x) where {T}
    return throw(
        ArgumentError(
            "Threaded particle evaluation requires every particle to have the same type, " *
            "got $(typeof(x)) after $T. Use SerialExecution for mixed particle types.",
        ),
    )
end

# Surface a worker's own exception so threaded and serial evaluation fail the same way.
function _wait_task(task::Task)
    try
        wait(task)
    catch e
        e isa TaskFailedException && throw(e.task.exception)
        rethrow()
    end
    return nothing
end

# Blocks take consecutive groups of four outputs from a SplitMix64 stream seeded by `key`,
# the seeding procedure recommended for xoshiro generators.
function _block_rng(key::UInt64, block::Integer)
    offset = 4 * (UInt64(block) - 1)
    s0, s1, s2, s3 = ntuple(j -> _mix64(key + (offset + j) * 0x9e3779b97f4a7c15), 4)
    return Random.Xoshiro(s0, s1, s2, s3)
end

function _mix64(z::UInt64)
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end
