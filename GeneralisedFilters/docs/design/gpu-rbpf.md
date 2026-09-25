# Shared CPU/GPU Rao–Blackwellised particle filtering

Status: implementation in progress, updated 2026-09-25. The representation,
initialisation and particle-storage foundation is implemented and tested. The
complete GPU filtering step and documentation benchmark are not yet implemented.
The original adversarial review informs the checks below.

### Implementation checkpoint

Implemented in the optional `BatchedKernelsExt` extension:

- Existing `initialise` dispatch on a device-backed hierarchical Gaussian prior,
  with `CUDA.RNG`, creates nested `BatchedStruct` particles with concrete floating
  weights and integer ancestry. CPU initialisation retains its existing route.
- Existing model methods return composite Gaussian atoms with explicitly shared
  or batched fields. Device Gaussian simulation accepts dense covariance or an
  existing `CovarianceFactor`; users can supply a root to avoid repeated factoring.
- Existing particle operations support eager structural gathering, exact integer
  ancestors, stable weight normalisation and the existing adaptive resampling path.
- Private raw Kalman helpers separate the unchanged numerical equations from CPU
  storage canonicalisation. The public CPU wrappers and AD cache remain intact.

No mandatory preparation object, population framework or new exported API has been
introduced. Initialisation selects storage through existing device-backed model
values, so an additional backend selector was unnecessary for this prototype.

Validation uses Julia 1.12.7 / CUDA 5.11.3 / RTX 4090. The original focused
runs passed the storage/model tests, 100 existing GPU regression/resampler assertions
and the selected CPU numerical, AD and type-stability suites. Test-environment
dependency errors were resolved and affected items rerun successfully. These are
focused checks, not a claim that the full repository suite was run.

Following review, the new tests use representative cases rather than precision ×
shape × ancestry grids. Storage coverage retains a non-warp-aligned population and
one duplicate/reordered ancestor map, ownership, integer exactness, adaptive
resampling branches, evidence and invalid/large-offset weights. Model coverage uses
Float32 dimension 16 plus a small Float64 case, fixed-noise draws, shared and
conditional beliefs, ownership and zero/rank-deficient covariance factors. CPU
coverage targets mixed-precision canonicalisation and the AD cache contract;
existing suites cover general numerical and derivative correctness.
The reduced tests pass: 107 GPU and 34 CPU assertions. The extension ambiguity
check also passes; formatting uses JuliaFormatter 2.14.0 with the repository style.

The remaining model `Device*` names are local aliases of existing parametric types,
not new model objects. Single-use aliases were inlined without changing dispatch.
Generic composite construction/reconstruction and recursive gathering are candidates
for BatchedKernels, whose current public interface lacks those helpers. Do not
replace this boilerplate with private BatchedKernels reconstruction functions.
Weight/evidence/ancestry semantics remain GeneralisedFilters responsibilities.

**Pending design decision:** the shared Kalman equations require BatchedKernels
support for a distinct-vector `dot`, traced division/scaling, identity construction
and the existing symmetrisation expression. Extending conventional BatchedKernels
operations is recommended over an internal numerical adapter, but neither route
has been implemented pending discussion. The sibling BatchedKernels checkout is
unchanged. A separate numerical probe passed Float32/Float64 checks through dimension
16; its adapter is not production code. Float64 dimension 16 exceeded the default
128-thread shared-memory budget in that probe; the existing 64-thread option worked.

An existing CPU normalisation edge case was also identified: five Float32 log
weights equal to `-1f8` lose the normalising offset. The GPU implementation uses
max-shifted normalisation and tests this case; changing the CPU implementation is
separate outstanding work. Neither this checkpoint nor the numerical probe establishes
end-to-end filtering correctness, transfer behaviour or a GPU speedup.

## 1. Thesis endpoint and scope

Demonstrate that one scalar Gaussian filtering computation and the existing
particle-filter recursion can execute over CPU particles or GPU composite batches.
The research claim is the composition of a model/algorithm interface, efficient
composite representation and generated batched numerical kernels, supported by
correctness and end-to-end performance evidence. A general GPU filtering framework
is not a prerequisite for this contribution.

The concrete endpoint is a documented bootstrap RBPF with Joseph covariance-form
Kalman updates, systematic resampling with an ESS threshold, final weighted state,
log evidence and filtering summaries. Use a dummy linear-Gaussian model for an
exact joint-Kalman reference and a stochastic-volatility factor model for genuinely
particle-dependent covariance work. Primary precision is Float32 and the target
inner dimension is up to 16. Treat outer and observation dimensions separately.

Host time iteration, scalar ESS/evidence work and synchronisation are allowed.
Main population calculations, sampling, gathering and weight operations run on the
GPU; no per-step particle-sized transfers to the host. Upload fixed inputs once;
small shared time-dependent inputs may be uploaded per step.

Essential for the first endpoint:

- Existing `RBPF(BF(N), KF())` semantics and shared resampling/evidence logic.
- CPU vector and nested GPU composite representations of the same logical states.
- One shared numerical definition for the supported Gaussian calculation.
- Explicit GPU initialisation, numeric weights, GPU noise generation and explicit
  shared versus batched model parameters for the two supported model families.
- Correctness tests and a reproducible CPU/GPU benchmark and docs example.

Deferred: mandatory preparation objects, a backend execution framework, a general
capability registry, arbitrary GPU callbacks/distributions, mixed-type integer
arithmetic in fusion, lazy resampling as a production requirement, buffer pools,
CPU threading, GPU AD, APF/CSMC/PGAS, smoothing and trajectory storage. Float64,
SRKF and ordinary PF are follow-ups when the core result is secure, not dependencies
of the first documented endpoint. Existing CPU features must continue to work.

## 2. Start with initialisation and existing dispatch

Do not introduce `prepare_filter` as a required lifecycle. The earlier draft grouped
allocation, type selection, root calculation and capability checking under that
name, but these do not require a separate prepared-filter object:

- Initialisation chooses CPU arrays or GPU batched storage and establishes types.
- Model construction or explicit upload wraps shared arrays and any common roots.
- Method dispatch selects operations from the actual storage and model types.
- Normal method-entry checks reject unsupported shapes/features when encountered.
- BatchedKernels caches compilation; changing numeric inputs must reuse it.

There must be one explicit choice of GPU storage at initialisation: dispatch cannot
infer that preference from an otherwise identical CPU model and algorithm. Choose
the smallest additive initialisation signature in the prototype. A keyword can
forward to a positional dispatch helper; Julia keyword arguments themselves are
not a dispatch axis. Do not pass new keywords into legacy user methods by default.

First demonstrate the existing manual `initialise`/`step` interface with a GPU
initial state. Then add a small convenience path to `filter` if needed to select
initial storage or accept an initial state, reusing its existing time loop. Exact
syntax is not frozen here. Do not introduce an execution context just to carry the
backend after storage already identifies it.

State storage alone does not make arbitrary model code device-compatible. The
initial supported models supply batched methods for outer sampling and conditional
model resolution, and device versions of their fixed arrays. This is an explicit
capability boundary, not an automatic compiler for all hierarchical models.

Preserve public CPU dispatch through `initialise`, `step`, `move`, `predict`,
`update` and particle hooks, including custom analytical filters inside RBPF.
Specialise GPU storage methods narrowly; never replace all `RBPF` moves by a fused
path merely because a nested algorithm is `KF`. Preserve observation validation,
`ref_state` forwarding and first-step behaviour on existing routes. Unsupported
GPU reference trajectories/algorithms must fail explicitly before modifying state.

### Reuse existing operations before adding names

| Need | Existing boundary / proposed minimal change |
| --- | --- |
| Extract log weights | Specialise `log_weights`; GPU result can expose the underlying device leaf without rebuilding particles |
| Normalised weights and ESS | Reuse `get_weights`, `will_resample`, and existing reduction helpers with device storage |
| Resampling and gather | Specialise `construct_new_state` and `preserve_sample`; retain `maybe_resample` ordering |
| Predict/update composition | Reuse `move`; use named scalar functions in map/broadcast where BatchedKernels requires them |
| Weight/evidence update | Reuse `marginalise!`, `add_logweight` and baseline semantics; specialise representation handling only |
| Field projection / reconstruction | Access composite components structurally; at most a small private helper if repeated code justifies it |
| Sampling | Retain `simulate` for CPU; add only the batch/noise operation required by supported GPU models |

Do not predeclare `population_states`, `population_logweights`, `population_ancestors`,
`map_population`, `weight_statistics`, or `assemble_population` as a new framework.
Use ordinary map/broadcast first. One small internal batch helper is acceptable if
it solves a demonstrated dispatch problem; specify that problem in its patch.
Avoid blanket overloads of `map` or `getproperty` on foreign types.

Independent maps, reductions/scans and gathers remain useful conceptual categories
for the thesis. They need not each become a new package API.

## 3. Composite storage, integers and resampling

Retain `ParticleDistribution`, `Particle`, `RBState` and `GaussianState` logically.
`ParticleDistribution` already permits `AbstractVector{<:Particle}`. Prototype:

```text
ParticleDistribution
  particles: BatchedStruct representing Particle
    state: BatchedStruct representing RBState
      x: batched outer vectors                   d_outer × N
      z: BatchedStruct representing GaussianState
        μ: batched means                         d_inner × N
        Σ: batched covariances                   d_inner × d_inner × N
    log_w: floating-point device vector           N
    ancestor: integer device vector               N
  ll_baseline: host scalar
```

Prove construction, concrete element types, zero-copy field extraction and
reconstruction before changing orchestration. Construction of a composite is
separate from passing that whole composite through tracing.

The current homogeneous floating-point fusion rule is an implementation restriction,
not a mathematical reason to exclude integer metadata. A dedicated source review
finds the restriction extends beyond one input check: output allocation and scalar
staging use a common T, and `TraceScalar{Int}` is a `Number`, not an `Integer`, so
it cannot replace the constrained ancestor type in `Particle` during reconstruction.

For the endpoint, the full population can keep exact integer ancestry while the
Gaussian fused call receives the `RBState`/Gaussian subcomposite and returns state
and likelihood increments. Reattach ancestor leaves by constructing host wrappers
around existing device arrays; that does not download particle data. Handle weight
updates outside that graph if necessary to preserve `Real`/typeless contracts.
Do not convert ancestors to floating point or weaken public constraints indiscriminately.

A narrowly scoped BatchedKernels improvement for integer pass-through is reasonable
if it simplifies composition enough to justify its implementation. It is distinct
from general integer arithmetic, mixed-precision matrix operations or dynamic gather
indexing, and is not on the endpoint's critical path. Tests must preserve exact
integer values above the Float32 exact-integer range, including composite outputs.

### Eager gather first; lazy gather is an optional experiment

Current CPU resampling constructs a new particle vector, but usually shares the
selected state's objects; it is not necessarily a deep copy of every covariance.
GPU structure-of-arrays gathering explicitly copies selected columns/slices. Use
out-of-place gathers initially, with duplicate, identity and permutation indices,
so source values and previously returned states remain valid. All state leaves
must use the same ancestor mapping; skipped resampling keeps weights and sets
identity ancestry. Shared model leaves remain shared.

Lazy gathering could represent a batch as a source plus ancestor indices and load
the selected parent when a kernel reads an input. Do not assume this needs compiler
changes: a narrow runtime probe demonstrated the existing fused loaders can read
an indexed GPU view. The ordinary `BatchedCuMatrix` constructor currently blocks
this: its view-element-type discovery dereferences a device ancestor index on the
host. The diagnostic bypassed that constructor with explicit type parameters;
this does not establish a valid general public container interface. A proper
constructor fix must discover the correct element type without reading device
indices. Remaining checks include vector/scalar leaves, wrapper combinations,
lifetimes and repeated-step index composition. See section 8 for test scope.

Retain eager gather unless measurements show it is important. Lazy access can save
an intermediate write but may repeat irregular reads across separate kernels and
retain old populations. New draws and output identity are indexed by child slot,
not parent, so duplicated parents must still receive independent innovations.
Do not alias writable outputs onto parent storage. Test covariance-byte traffic
and actual elapsed gather time rather than assuming either approach wins.

## 4. Explicit sharing within the hierarchical SSM interface

Keep the scalar model meaning unchanged:

```julia
inner_dynamics(component, t, x_prev, x_new) -> LinearGaussianDynamics(A, b, Q)
inner_observation(component, t, x) -> LinearGaussianObservation(H, c, R)
```

A model can additionally supply a batch-aware method for those same generics when
outer arguments are batched containers. It returns a batch of ordinary model atoms,
with each field explicitly batched or shared. No global capability registry is
needed: a supported method is the capability.

For example, particle-dependent A and shared Q can be represented schematically as:

```text
BatchedStruct representing LinearGaussianDynamics
  A = BatchedCuMatrix(A_device)       # d × d × N
  b = BatchedCuVector(b_device)       # d × N (or SharedCuVector)
  Q = SharedCuMatrix(Q_device, N)     # one d × d matrix
```

These wrappers already exist. To make the prototype concrete, the equivalent
explicit construction is:

```julia
fields = (; A=A_batch, b=b_batch, Q=Q_shared)
D = LinearGaussianDynamics{eltype(A_batch), eltype(b_batch), eltype(Q_shared)}
dynamics_batch = BatchedStruct{D,typeof(fields)}(fields, N)
```

This is a batch *of* `LinearGaussianDynamics`, not one atom whose A happens to be
a vector of matrices. The latter violates today's `A<:AbstractMatrix` atom contract.
Likewise, the generic scalar `inner_dynamics` currently validates its result as a
`LatentDynamics`; a `BatchedStruct` is not one. Add a narrowly dispatched batch path
and validate its element type, fields and batch sizes there. Do not broadly weaken
scalar validation or claim the existing generic path accepts batched results today.
The scalar Gaussian function then receives one reconstructed atom while tracing,
so the numerical/model meaning remains common.

Constant atoms already explicitly declare particle independence. A batch method
can wrap their fields as shared inputs automatically. If a callback varies A while
Q is common, the model author states that through the returned field wrappers.
For the two initial models, provide short example batch methods rather than a
system that analyses arbitrary closures or compares resulting arrays for equality.

A fill array or `Ref(Q)` is a sensible higher-level spelling of sharing. Current
BatchedKernels input handling does not automatically translate arbitrary Ref/fill
objects into shared device arrays. Start with `SharedCuMatrix`/`SharedCuVector`;
add conversion syntax only if it materially improves the example. `fill(Q,N)`
means repeated references to Q in Julia, not an automatic device shared-input mode.
Changing runtime arrays must not be encoded as `SharedValue` literals in compile keys.

Outer sampling uses bulk device standard normals and a deterministic transition
transform. CPU sampling can use the same transform with CPU draws. Do not require
arbitrary `Distributions.jl` sampling inside a kernel or assume equal seeds generate
equal CPU/GPU draws. The GPU path must not generate particle-sized noise on the CPU.

For the SV example choose stable A and Φ and specified Gaussian priors:

```text
h_t = a + Φ*(h_{t-1}-a) + L_h*ε_t
z_t = A*z_{t-1} + diag(exp(h_t/2))*η_t
y_t = H*z_t + e_t,       e_t ~ N(0,R)
```

The noises are independent and R is positive definite. The batch resolver computes
Q_t=diag(exp(h_t)) on device; current tracing lacks the needed exp/diagonal route,
so use a simple device broadcast/kernel before the shared Gaussian update. Further
fusion is optional. The dummy model uses x_prev in its inner drift; test both time
conventions. Moderate stationary volatilities avoid overflow in the default fixture.

## 5. Shared numerical core and probability contracts

Share the supported Gaussian equations through small GeneralisedFilters-owned
numerical functions. Keep host canonicalisation (`_kalman_state` converts generic
arrays to Vector/Matrix) outside tracing. Preserve the existing CPU reverse cache,
repair policies and AD behaviour. Do not add a second maintained Joseph formula
by copying the BatchedKernels example into an extension.

Probe the actual intended core before broad CPU refactoring:

- `one(S)` has no current traced identity construction: supply a shared identity or
  a small semantic helper using public primitives.
- Generic `symmetrise` divides a matrix by two; use the supported semantic helper.
- Traced scalar division by two can use equivalent type-correct scaling.
- Verify Cholesky/solve, dot, logdet and wrapper combinations, not just adapter parity.
- Preserve cached intermediate meaning; any algebra/adjoint change is separately
  validated. Prefer a local extraction over redesigning all CPU analytical filters.

Expose the existing combined `move` composition to fusion for supported storage,
while retaining separate predict/update interfaces. The numerical kernel may consume
already-resolved batched model atoms; it need not fuse the entire model/sampler.
Use named functions and public BatchedKernels APIs. Do not depend on private tracing
classes, compiler caches or launch internals from GeneralisedFilters.

Initial support is `KF(repair=NoRepair())`. Reject unsupported GPU repair policies
and shape combinations explicitly. Test the actual (outer, inner, observation)
tuples through 16 and partial batches. Individual primitive block extents and GPU
resources constrain the graph; implicit QR stacks, if later used, may exceed the
32-lane extent logically and must not be rejected by an incorrect blanket rule.

### Initial weights do not require preparation

GPU initialisation explicitly chooses the supported floating-point calculation type
and sets numeric zero log weights, identity/initial ancestry as required, and the
matching baseline. Reject mismatched model/input types at the relevant boundary.
The exact bootstrap proposal correction can be handled as zero without entering
the trace as a typeless marker. Legacy CPU initialisation retains `TypelessZero`,
`TypelessBaseline` and first-density type promotion; do not change it globally.

For ordinary filtering, after deciding whether to resample:

```text
b = logsumexp(incoming_logweights)
v_i = incoming_logweight_i + proposal_correction_i + observation_increment_i
a = logsumexp(v)
log_evidence_increment = a - b
new_logweight_i = v_i - a
```

Compute b before corrections. Bootstrap corrections are zero. Reset weights to
zero on resampling, giving b=log(N); preserve normalised weights on skipped steps.
Preserve the existing initial-uniform resampling law/threshold rather than silently
skipping it. APF's modified baseline stays on its existing CPU route. N remains an
integer; test count logarithms rather than assuming exact conversion to Float32.

Use stable log-sum-exp/ESS, with host scalar results allowed. As an initial accuracy
baseline use Float64 CDF accumulation on CPU/GPU when comparing systematic resampling;
state/log-weight storage can remain Float32. Measure this choice rather than treating
it as permanent. Compare Float32 scans against high-precision CDF/offspring references
before recommending them for a stated N range. Promoting a scan cannot recover mass
already underflowed in input probabilities. Record collective precision in benchmarks.

Current GPU offspring expansion can assign nearly all N descendants to one parent's
thread. Include concentrated weights in stage profiling. Only replace this with
parallel inverse-CDF/expansion work if needed, preserving the systematic law.

For supported Gaussian inputs, verify finite state/likelihood outputs and report
invalid calculations with a time index. Current traced Cholesky's literal success
code is not runtime status. An output check alone does not certify covariance PSD;
validate shared covariances once and document the failure-detection limits. Include
one invalid particle among valid ones in testing. A checked-factorisation primitive
is a separate BatchedKernels improvement if the supported error contract needs it;
a comprehensive diagnostic subsystem is not required before a valid-model prototype.
Individual -Inf density weights may be valid, but all-impossible weights, NaNs and
+Inf must not silently produce a normalised NaN cloud. Do not mask failures with jitter.

## 6. Implementation sequence with stopping points

### A. Small representation/dispatch experiment

Construct CPU and GPU initial populations with numeric GPU weights. Demonstrate
field access through existing helpers, batched atoms with shared/batched fields,
one named Gaussian operation on nested composites, and repeated output reuse.
Probe eager gather and optionally one indexed-view lazy gather. Verify integer
ancestry remains exact without requiring whole-particle fusion. Compile the target
shapes early. Freeze only the small helpers actually needed by this experiment.

Acceptance: public APIs only, no per-particle host access, valid concrete container
types and retained old states unchanged. No `prepare_filter` or population framework
is introduced. If a representation fails, record the exact obstacle before adding
an adapter; do not solve it by converting ancestors to floats.

### B. One complete bootstrap RBPF

Extract the shared Gaussian core and connect the two example models' explicit
batch methods, device draws, existing resampling orchestration, eager gather,
weight normalisation and evidence. Preserve default CPU routes and custom dispatch.
Keep generic callbacks on CPU; unsupported GPU models get an actionable method/error.

Acceptance: fixed-noise/ancestor multi-step comparisons, exact-model statistical
checks, both ESS branches and a transfer profile showing no N-sized host traffic
per step. CPU tests for numerical/AD changes and affected particle dispatch pass.
Sentinel custom initialise/step/move/predict/update and particle methods verify
unchanged call order/forwarding before and after any shared-loop refactor.

### C. Thesis experiment and documentation

Produce the runnable example, raw timings, accuracy results and plots. Profile
before optimising allocations, resampling or gathering. Public `fuse` allocations
and scalar CPU decisions are acceptable. Finish this endpoint before broadening
algorithms, samplers or compiler capabilities.

Acceptance: reproducible CPU/GPU comparison including small-N losses/crossover,
correctness evidence and a docs build that works without a GPU. No speedup number
is required to pass; report actual results and limitations.

## 7. Validation and benchmark requirements

- Supply identical initial samples, noise and ancestor maps for deterministic
  CPU/GPU comparisons of states, covariances, weights, increments and total evidence.
  Fix resampling schedules in these checks; floating-point differences can change
  adaptive decisions. Test actual adaptive branching separately.
- Compare repeated complete runs to the joint Kalman reference using Monte Carlo
  uncertainty. Likelihood estimates, not log estimates, are the unbiased quantity;
  use stable likelihood-domain aggregation when testing evidence.
- Mixture covariance includes within-particle covariance and variance of conditional
  means. Compute summaries on device and download small results, not full clouds
  each time. Full trajectory/history storage is out of scope.
- Cover N=1 and partial batches (e.g. 5,31,32,33), identity/duplicate/permuted ancestors,
  zero-interspersed/concentrated weights, initial uniform resampling, endpoint/CDF
  rounding and no invalid ancestor indices or missing descendants. Verify resampling
  precision against high-precision references over the benchmark range.
- Cover dimensions 1,3,8,16 with explicit outer/observation sizes, valid singular
  process noise, positive observation noise, nonfinite/invalid inputs, type mismatch,
  time-dependent parameters and repeated calls without timestep-specific compilation.
- Disable GPU scalar indexing except intentional audited scalar reads; also profile
  transfers because that setting does not detect explicit array copies.
- Preserve snapshots/ownership and run affected CPU StaticArrays/dynamic, weight,
  PF/RBPF/APF/CSMC and AD regressions; run the full CPU suite before integration.
  Real GPU tests are separate from extension-load checks.

Benchmark existing CPU StaticArrays execution and the shared-core CPU/GPU path for
the same model, precision, resampling policy and requested outputs. Use stable
transition parameters: the dummy generator's random matrix need not be stable at
D=16. The dummy model permits shared covariance evolution; disclose redundant work
and do not base a best-performance claim on ignoring this structure. Use the SV
model as the main particle-dependent workload.

Sweep roughly N=10^2 to 10^5 and inner dimensions 1,3,8,16, subject to memory limits;
a stable horizon such as T=500 is a starting point. Report stage costs, including
uniform and concentrated resampling/gather cases, as well as complete warmed calls
with GPU completion. Include sampling, per-step model construction, reductions,
resampling and summaries. Report initialisation, compilation, upload and final
extraction separately, plus inclusive time-to-result. Reset state per measurement.
Record raw samples, seeds, precisions, hardware, Julia/dependency revisions and
thread counts. Do not turn asynchronous launch latency into an inference timing.

Use an optional extension activated by CUDA and BatchedKernels; no new mandatory
GPU dependency. Reuse CUDA resampling through GeneralisedFilters functions rather
than extension-module internals. Start from the tested Julia 1.12.7/CUDA 5 environment;
test other versions before claiming them. Pin the actual BatchedKernels working
state reproducibly before publishing results, without committing machine-local paths.

The Literate docs build currently executes discovered examples on CPU-only runners.
Provide an explicit optional GPU section and recorded hardware-labelled results, or
consume artifacts generated on a GPU. Never label a CPU fallback as GPU execution.
Do not make ordinary package tests or docs require a GPU.

## 8. Evidence, review history and remaining questions

GeneralisedFilters was reviewed at `3fd71e5`. The BatchedKernels worktree is based
on `ac9905e` but contains substantial uncommitted changes; that commit alone is not
a reproducible identifier for the reviewed code. The existing integration harness
passed 1,080 checks on Julia 1.12.7/CUDA 5.11.3/RTX 4090 in the preceding review.
This does not prove complete GPU filtering or traceability of the current GF core.

The first independent adversarial review identified six substantive issues:
custom dispatch preservation, actual Joseph traceability, CDF precision, concentrated
resampling performance, numerical failure status and early shape/SV primitive checks.
A second pass found those addressed in the previous draft. This scoped revision
retains the checks but removes mandatory preparation, capability registration and
speculative population APIs in response to the user's thesis timescale and feedback.

A separate agent (`integer_fusion_review`) inspected integer fusion and eager/lazy
gather. Integer exclusion is an implementation restriction involving input checks,
per-leaf allocation/staging and constrained composite reconstruction; removing the
input check alone is insufficient. No mixed-integer implementation was attempted.

Its indexed-view runtime probe passed 12/12 diagnostic checks on Julia 1.12.7 and
CUDA.jl 5.11.3, with scalar indexing disabled: Float32 fused matrix multiplication
at square dimensions 3 and 16 and rectangular dimensions (3,4,2), using 19 offspring
from 23 parents and duplicate/permuted Int32 device indices. Lazy results matched
CPU references and eager gathering; parent storage and indices were unchanged.
The script is `/tmp/gf-indexed-view-smoke.jl` (a local investigation artifact).

The ordinary constructor failed as described in section 3. The diagnostic supplied
an explicit element-view type derived from the original parent and did not establish
that it equals the nested indexed view's actual element type. The result therefore
validates loader feasibility only, not public lazy-gather support or a full filter.
No repository source was changed and no performance claim follows from this test.
Lazy gathering is now a promising small constructor/storage experiment, with eager
gathering retained as the endpoint fallback.

Questions to settle through the small prototype, not a framework design phase:

1. What is the smallest initialisation storage selector, and does the docs endpoint
   need any additional top-level `filter` convenience at all?
2. Which narrowly dispatched batch methods for `inner_prior`, `inner_dynamics` and
   `inner_observation` give the clearest example with explicit sharing?
3. Does ordinary named-function broadcast plus structural component access suffice,
   or is one small batch helper genuinely required?
4. Can the shared Gaussian core preserve the CPU cache/AD contract with local
   helper changes, and what numerical failure detection can it actually provide?

These questions do not require solving arbitrary GPU model compilation, complete
integer arithmetic or optimal resampling to establish the initial thesis result.
