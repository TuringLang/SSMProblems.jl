# GPU Rao–Blackwellised filtering

The experimental BatchedKernels extension uses the same Kalman equations and
conditional-state prediction/update recipe and particle-filter orchestration as
CPU filtering. Gaussian states and model atoms
are represented by nested `BatchedStruct`s. The extension broadcasts the same
`kalman_predict` and `kalman_update` functions used on the CPU; BatchedKernels
fuses their numerical operations into GPU kernels. Time iteration and scalar reductions
remain host-controlled; particle sampling, Gaussian updates, weights and ancestor
gathering operate on the GPU.

This example requires the local BatchedKernels development version with traced
`one`, division, matrix-plus-adjoint and distinct-vector `dot` support, plus the
`BatchedStruct(Type, components)` constructor. It is not
supported by an arbitrary released version of BatchedKernels. A CUDA GPU is required;
the ordinary documentation build does not execute this example.

## Model and execution

The runnable model is `examples/gpu-rbpf/model.jl` in the GeneralisedFilters
package directory.
Activate an environment containing GeneralisedFilters, BatchedKernels, CUDA and
StaticArrays, with both development packages selected using `Pkg.develop(path=...)`.

```julia
include("examples/gpu-rbpf/model.jl")
using .GPUVolatilityExample, GeneralisedFilters, CUDA, Random
CUDA.allowscalar(false)
cpu_model, gpu_model = GPUVolatilityExample.models(16, 4, Float32)
ys = GPUVolatilityExample.observations(cpu_model, 20)
device_ys = [CuArray(Vector(y)) for y in ys]
algo = RBPF(BF(8192; threshold=0.5), KF())
state, log_evidence = GeneralisedFilters.filter(CUDA.RNG(1), gpu_model, algo, device_ys)
mean = GPUVolatilityExample.inner_mean(state)
```

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

## Supported boundary

The initial route supports bootstrap RBPF with a covariance-form Kalman filter,
`NoRepair`, device Gaussian outer sampling, explicit batch model methods and device
observation vectors. It returns the final particle population and log evidence.
Reference trajectories, GPU AD, smoothing and arbitrary GPU samplers are outside
this example. Filtering summaries can be reduced on device as shown above.

Models must provide valid covariance matrices. A finiteness check rejects
non-finite likelihoods if the generated kernel returns, but Cholesky can instead
fail inside the CUDA kernel before that check runs. The failure contract is under
review; this route does not provide the CPU factorisation's runtime exception
contract or a full positive-semidefiniteness certificate. Float32 through
inner dimension 16 is the primary target; larger shapes and Float64 require
separate shared-memory and accuracy checks.
