# GPU Rao–Blackwellised filtering

The experimental BatchedKernels extension uses the same Kalman equations and
conditional-state prediction/update recipe and particle-filter orchestration as
CPU filtering. Gaussian states and model atoms
are represented by nested `BatchedStruct`s. The extension broadcasts the same
`kalman_predict` and `kalman_update` functions used on the CPU; BatchedKernels
fuses their numerical operations into GPU kernels. Time iteration and scalar reductions
remain host-controlled; particle sampling, Gaussian updates, weights and ancestor
gathering operate on the GPU.

This example requires the current GeneralisedFilters development checkout,
BatchedKernels 0.2.2 or later in the 0.2 series, and a CUDA GPU.
The ordinary documentation build does not execute GPU examples.

## Model and execution

The runnable model is `examples/gpu-rbpf/model.jl` in the GeneralisedFilters
package directory.
Activate an environment containing GeneralisedFilters, BatchedKernels, CUDA and
StaticArrays, with both development packages selected using `Pkg.develop(path=...)`.

```julia
include("examples/gpu-rbpf/model.jl")
using .GPUVolatilityExample, GeneralisedFilters, CUDA, BatchedKernels, Random
CUDA.allowscalar(false)
cpu_model, gpu_model = GPUVolatilityExample.models(16, 4, Float32)
ys = GPUVolatilityExample.observations(cpu_model, 20)
device_ys = [CuArray(Vector(y)) for y in ys]
algo = RBPF(BF(8192; threshold=0.5), KF())
rng = CombinedRNG(Xoshiro(1), BatchedRNG(2))
state, log_evidence = GeneralisedFilters.filter(rng, gpu_model, algo, device_ys)
mean = GPUVolatilityExample.inner_mean(state)
```

`CombinedRNG` bundles the caller-owned CPU and GPU generators into the ordinary
RNG argument. Fused particle initialisation and transitions use its GPU child.
`rand!(rng, device_array)` and `randn!(rng, device_array)` also use that child;
ordinary CPU arrays and scalar draws use its CPU child. BatchedRNG currently
supports dense Float32/Float64 CuArray destinations; unsupported device views
raise an error instead of falling back to CPU draws. Multinomial and stratified
resampling generate uniform arrays on the GPU, while systematic and conditional
reference offsets use host scalar draws. Both children advance across calls, and
both must be saved to reproduce continuation. Copying the bundle copies both
streams; it does not split them into independent streams. `Random.seed!(rng, seed)`
resets both children, deriving the GPU seed from a separately seeded CPU copy.
Use an explicit CPU generator such as `Xoshiro`; `TaskLocalRNG` is unsupported
because it does not provide independently owned checkpoint state. The original
single-`CUDA.RNG` route remains supported. The older `filter_gpu` example helper
is available for callers managing particle and resampling generators separately.

The outer state is scalar log volatility. Conditional on it, a 16-dimensional
Gaussian factor process has process covariance `exp(x_t) * Q`. Each particle thus
has its own covariance evolution. The observation dimension is four. Fixed model
arrays and observations are uploaded before filtering; only the final mean is
downloaded in this example.

`VolatilityDynamics` defines an ordinary conditional CPU model and an explicit
`inner_dynamics` batch method. That method returns a `BatchedStruct` of the existing
`LinearGaussianDynamics` type, with `SharedCuMatrix`/`SharedCuVector` for constant
`A` and `b` and a `BatchedCuMatrix` for particle-dependent `Q`. Sharing is declared
by storage, never inferred by comparing values. Initialisation dispatches on the
device-backed outer Gaussian prior; there is no preparation object.

## Reference trajectories

Pass the existing `ref_state` keyword to fix particle 1's outer trajectory.
Include the initial state at time zero. Upload the states once; each state must
be a device vector with the outer state's dimension and precision. Device views
into an uploaded trajectory matrix are also supported.

```julia
# A constant reference for the example's scalar outer state.
reference = ReferenceTrajectory(
    CuArray(Float32[0]), [CuArray(Float32[0]) for _ in ys]
)
state, log_evidence = GeneralisedFilters.filter(
    rng, gpu_model, algo, device_ys; ref_state=reference
)
```

The single-`CUDA.RNG` route through `GeneralisedFilters.filter` accepts the same
keyword. Conditional multinomial, systematic and stratified resampling reuse the
existing CUDA implementations. When ESS skips resampling, particles retain their
weights and use identity ancestry. The reference fixes only the outer state: its Gaussian
belief is still predicted and updated from the selected ancestor.

Sampling currently draws the full batch before replacing particle 1, so
random-stream consumption need not match the CPU implementation.

`ConditionalSMC` with `NoRefreshment()` uses the same sampling loop as on the CPU,
with dense GPU history. The returned reference contains compact device vectors:

```julia
using AbstractMCMC
sampler = ConditionalSMC(algo)  # NoRefreshment()
model = CSMCModel(gpu_model, device_ys)
rng = CombinedRNG(Xoshiro(3), BatchedRNG(4))
sample, sampler_state = AbstractMCMC.step(rng, model, sampler)
sample, sampler_state = AbstractMCMC.step(rng, model, sampler, sampler_state)
trajectory = sample.trajectory  # indexed from 0, outer states only
```

For manual filtering loops, `DenseParticleContainer(initial, first_state)` and
`push!(history, state)` retain populations on device; `get_ancestry(history, i)`
extracts one path. Dense history costs O(NT) storage. Ancestry tracing transfers
individual indices to the CPU, without downloading particle populations.
Use `ConditionalSMC(algo, AncestorSampling())` or
`ConditionalSMC(algo, BackwardSimulation())` for trajectory refreshment. These use
the default square-root Gaussian backward predictor, sharing the CPU time loops
and backward-weight formula. Population weights run on the GPU; selected-path
likelihood updates use batches of size one. Both strategies currently retain dense
history. The outer Gaussian transition must have a nonsingular covariance for its
density to be defined.

To smooth the inner Gaussian process conditional on a sampled outer trajectory,
use the ordinary conditional-model and Kalman-smoother interface:

```julia
conditional_model = condition_inner(gpu_model, trajectory)
smoothed, conditional_log_evidence = GeneralisedFilters.smooth(
    rng, conditional_model, KS, device_ys; t_smooth=1
)
mean_at_first_observation = Array(smoothed.μ)
```

This returns the conditional Gaussian marginal at `t_smooth`; its mean and covariance
remain on device. Model resolution and Gaussian calculations use batches of size one,
while the same smoothing time loop serves CPU and GPU models.

## Supported boundary

The initial route supports bootstrap RBPF with a covariance-form Kalman filter,
`NoRepair`, device Gaussian outer sampling, explicit batch model methods and device
observation vectors. It returns the final particle population and log evidence.
The `SerialExecution` and `ThreadedExecution` settings control CPU population traversal;
device-backed populations use BatchedKernels regardless of that setting.
GPU AD and arbitrary GPU samplers are outside this example. Filtering summaries can be reduced on device as shown above.

Models must provide valid covariance matrices. A finiteness check rejects
non-finite likelihoods if the generated kernel returns, but Cholesky can instead
fail inside the CUDA kernel before that check runs. The failure contract is under
review; this route does not provide the CPU factorisation's runtime exception
contract or a full positive-semidefiniteness certificate. Float32 through
inner dimension 16 is the primary target; larger shapes and Float64 require
separate shared-memory and accuracy checks.

## Population ownership and execution

The bootstrap/Kalman RBPF path shares population orchestration between ordinary
CPU particle arrays and `BatchedStruct` populations. Internal
`_rb_population_fields` / `_assemble_rb_population` adapters separate storage from
execution. Field extraction supplies borrowed inputs, not a writable projection:
replacing an entry in a collected CPU field does not replace the original particle.
Assembly creates a new population representation and borrows the supplied leaves;
it does not deep-copy state or metadata. History preservation remains responsible
for obtaining independent storage.

CPU execution retains `predict_particle` / `update_particle` hooks and the existing
serial or threaded RNG traversal. The device adapter evaluates the same RBPF
state recipe on whole field batches. Reference conditioning returns the prescribed
outer state directly on CPU; device propagation samples its batch and copies the
reference into the new outer-state member with `x[1] = reference`. Both happen
before resolving inner dynamics from the old and new outer states. Neither path
replaces the gathered old state or skips prediction of the Gaussian belief.

BK member assignment performs ordinary backing-array conversion, but the GF
reference adapter still requires a device vector matching the model precision.
Ancestry remains integer storage outside floating-point fusion. Custom model
adapters may lift a common per-particle numerical function with BK broadcast;
arbitrary CPU callbacks are not automatically GPU-compatible.
