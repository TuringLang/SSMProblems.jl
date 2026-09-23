# Migrating from 0.4.2 to 0.5

Version 0.5 replaces the model interface used in 0.4.2. This guide describes the
changes needed to update an existing model or inference script. If you are new to
the package, start with the [model guide](models/linear-gaussian.md) instead.

## Dependencies

GeneralisedFilters 0.5 requires Julia 1.12.7 or later and SSMProblems 0.7. Its Turing
integration supports Turing 0.47–0.49 and DynamicPPL 0.42.11 or later within 0.42.
Reverse-mode differentiation uses Mooncake 0.5.53 or later within 0.5.

Turing 0.46 depends on SSMProblems 0.6 through AdvancedPS and cannot be used with
this release. GeneralisedFilters provides its own particle Gibbs integration with
Turing. Models written for AdvancedPS's state-space integration need to be adapted
to this interface.

## Model definitions

GeneralisedFilters now re-exports the process types, model container and generic
functions from SSMProblems 0.7. Define model processes using these shared types.
GeneralisedFilters supplies the Gaussian model components, conditioning operations and
inference algorithms.

| 0.4.2 interface | 0.5 interface |
|:--|:--|
| Parameterised process types and methods with forwarded keywords | Unparameterised process types and methods without forwarded keywords |
| `HomogeneousGaussianPrior` | `GaussianPrior(μ, Σ)` |
| `HomogeneousLinearGaussianLatentDynamics` | `LinearGaussianDynamics(A, b, Q)` |
| `HomogeneousLinearGaussianObservationProcess` | `LinearGaussianObservation(H, c, R)` |
| Per-field `calc_*` methods | Closures that return a whole Gaussian component from a context |
| `MvNormal`/`PDMat` filtering states | `GaussianState`, accessed with `mean` and `cov` |
| Separate materialised Kalman likelihood | `marginal_loglikelihood(condition_inner(model, xs), KF(), ys)` |
| `KalmanFilter(jitter=ε)` | `KalmanFilter(repair=Jitter(ε))` |
| Callback-based history collection | An explicit `initialise`/`step` loop or CSMC history storage |
| RB references containing stored Gaussian beliefs | Outer-only `ReferenceTrajectory` objects |

For time-varying or parameter-dependent Gaussian processes, move the construction
of their matrices and offsets into a closure. The closure receives the time index
and any outer states needed by that process. See the [model guide](models/linear-gaussian.md)
for the context fields and examples.

Filtering results use `GaussianState` rather than a distribution with a cached
factorisation. Use `mean(state)` and `cov(state)` to inspect a result, or
`MvNormal(state)` when a distribution is needed. Structured covariance parameters
can still be supplied to Gaussian model components. The filter converts them to dense or
static covariance storage for its calculations.

`filter` and `marginal_loglikelihood` require at least one observation. Empty inputs
raise an `ArgumentError`, including when called through a conditional inner model.
Use `initialise` if you only need the initial filtering state.

## Conditional SMC and particle Gibbs

Store only the outer trajectory in an RB particle Gibbs reference. Inner Gaussian
beliefs are recomputed on each sweep, so a reference remains usable after model
parameters change.

Conditional SMC now requires a resampler with an implemented conditional law.
`Multinomial()`, `Systematic()` and `Stratified()` are supported. `Metropolis()`
and `Rejection()` remain available for ordinary particle filtering, but cannot be
used with conditional SMC. Custom resamplers must implement
`conditional_sample_ancestors` and opt in through `supports_conditional`.

The conditional sampler draws all offspring subject to the reference constraint.
It no longer draws unconditionally and overwrites a single index, which does not
produce the required law for dependent resampling schemes.

Ancestor sampling respects the ESS threshold: it refreshes the reference ancestor
when the particle population is resampled. When resampling is skipped, ancestors
and accumulated filtering weights are retained. There is currently no option to
refresh the reference ancestor at those skipped steps.

`AuxiliaryParticleFilter` now supports `NoRefreshment()`, `AncestorSampling()` and
`BackwardSimulation()`, including when wrapping an RBPF. Lookahead weights are
accounted for in ancestor selection and filtering-weight corrections.

## Covariance handling and backward sampling

For RB ancestor sampling and backward simulation, `KF()` and `SRKF()` now use
`SqrtBackwardInformationPredictor()` by default. It represents backward
likelihoods as factored Gaussian residuals and evaluates them with QR
factorisations and triangular solves. The older `BackwardInformationPredictor()`
remains available, with its scalar precision-cancellation defect corrected.

Use `SRKF()` if you also want square-root forward filtering.
`CovarianceFactor(F)` supplies a covariance as `F * F'`. Prior and process factors
may be rectangular or rank deficient. Covariances supplied as ordinary matrices
still need a Cholesky factorisation on this route, and observation noise must be
positive definite. Support for a singular filtering covariance does not imply
support for a full-dimensional Gaussian `logpdf` of that state.

Exact RB ancestor sampling and backward simulation reject filtering-state
covariance repair and nonzero backward jitter. Their backward likelihoods must
agree with the forward model. If regularisation is needed, include it in the
model's covariance parameters so that filtering, backward sampling, simulation and
parameter updates all use the same model. Square-root filtering can improve
numerical stability without changing these covariances.

## Loading packages

Use Julia's normal precompiled package loading. With the supported DynamicPPL
version, source loading under Julia 1.12 with `--compiled-modules=no` encounters
an upstream generated-function binding error.

## Custom particle updates

Use `add_logweight(old_weight, increment)` when extending particle-level weight updates.
The initial zero-weight marker is internal bookkeeping and no longer supports generic
numeric conversion or arithmetic. Ordinary density and proposal methods continue to return
numeric log-density contributions.

Particle weights may acquire their numeric type during the first complete step. Later
steps must preserve that type. CSMC history now retains the weight precision instead of
converting weights to Float64, and rejects incompatible subsequent weight types.

## Recording histories without callbacks

Use `ParticleTree(initial)` when initial and later state types match, or
`ParticleTree(initial, first_state)` to infer the later type from the first completed step.
`DenseParticleContainer(initial, first_state)` infers both state and weight types.
Append subsequent results with `push!(container, state)`.

Containers copy collection buffers but share the state objects inside them. Custom
transitions and updates used with history storage must not mutate retained states. Use
explicit snapshots where necessary. See [Recording filtering results](history.md) for
an executable loop and ownership guidance.
