# Multithreading

Particle filters can evaluate their particles on several threads. Start Julia with more
than one thread, for example `julia --threads=auto`, and select `ThreadedExecution` for the
particle filter:

```julia
pf = BF(10_000; threshold=0.5, execution=ThreadedExecution())
state, loglikelihood = GeneralisedFilters.filter(rng, model, pf, observations)
```

Filters built from a particle filter use its setting. For example,
`RBPF(BF(1_000; execution=ThreadedExecution(blocksize=8)), KF())` evaluates its
Rao–Blackwellised particles in parallel. Wrapping it in `AuxiliaryParticleFilter` also
parallelises the lookahead, and wrapping it in `ConditionalSMC` also parallelises the
`AncestorSampling()` and `BackwardSimulation()` weights. The default,
`SerialExecution()`, evaluates particles in order on the calling task.

## What runs in parallel

Initialisation, prediction, observation updates, auxiliary lookahead weights and ancestor
sampling or backward simulation weights are evaluated in parallel. Resampling, weight
normalisation and particle history storage remain serial. The speed-up is therefore
largest when each particle is expensive to update, as for Rao–Blackwellised particles or
costly nonlinear models. When each particle update takes only a few nanoseconds, the
serial steps limit the gain.

## Reproducibility

Particles are evaluated in consecutive blocks of `blocksize` particles, 32 by default. Each
block draws from its own random number generator, derived from a single draw of the
generator passed to the filter. For a given generator state, number of particles and
`blocksize`, results are the same for any number of threads or tasks. Changing
`blocksize` changes the random draws. Threaded and serial runs use different random
streams, so their results differ, but calculations that draw no randomness, such as
observation updates and ancestor weights, give identical values.

The block size is also the smallest amount of work given to a task. Prefer larger blocks,
such as 1024, when each particle is cheap, and smaller blocks, such as 4 to 32, for
Rao–Blackwellised particles. `ThreadedExecution(ntasks=4)` limits the number of tasks
without changing the results.

## Requirements on model code

With threaded execution, model code runs concurrently for different particles:

- Draw randomness only from the generator passed to `simulate` or a proposal, never from a
  captured or global generator.
- Do not mutate state shared between particles, such as a captured buffer or cache.
- Every particle must have the same concrete type. Threaded evaluation throws an
  `ArgumentError` otherwise.

Allocation in model code limits scaling, because threads contend for the garbage
collector. StaticArrays avoid this for small states. Models using BLAS on heap arrays
should usually call `LinearAlgebra.BLAS.set_num_threads(1)` to avoid oversubscribing the
processor.

## Automatic differentiation

ForwardDiff differentiates through threaded evaluation. Reverse-mode differentiation with
Mooncake does not support threads and raises an error: use `SerialExecution()` for
particle objectives differentiated in reverse mode. Particle Gibbs does not differentiate
through the particle filter, so its CSMC sampler can use threaded execution with either
backend.

## Independent filters

Separate filters or chains can also run concurrently. AbstractMCMC's `MCMCThreads()` runs
several chains in parallel, and independent calls to `filter` can run in separate tasks
if each receives its own generator. Threaded execution within a filter uses Julia tasks,
so it can be combined with these approaches.
