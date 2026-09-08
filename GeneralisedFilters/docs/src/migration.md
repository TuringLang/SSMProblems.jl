# Migrating to 0.6

Version 0.6 changes the model interface and therefore is not a backwards-compatible patch
release of 0.5.

This research release targets current stable dependencies: Julia 1.12.7 or later,
Turing 0.46.1 or later within 0.46, DynamicPPL 0.42.11 or later within 0.42, and Mooncake 0.5.53 or later
within 0.5. Older Julia and Turing interfaces are no longer supported.

| Previous interface | 0.6 interface |
|:--|:--|
| SSMProblems process types/generics | GeneralisedFilters process types/generics |
| `HomogeneousGaussianPrior` | `GaussianPrior(μ, Σ)` |
| `HomogeneousLinearGaussianLatentDynamics` | `LinearGaussianDynamics(A, b, Q)` |
| `HomogeneousLinearGaussianObservationProcess` | `LinearGaussianObservation(H, c, R)` |
| Per-field `calc_*` methods and forwarded keywords | Closures returning whole atoms from an explicit context |
| MvNormal/PDMat filtering states | `GaussianState`, with `mean`, `cov` and optional `MvNormal(state)` conversion |
| Separate materialised Kalman likelihood | `marginal_loglikelihood(condition_inner(model, xs), KF(), ys)` |
| `KalmanFilter(jitter=ε)` | `KalmanFilter(repair=Jitter(ε))` |
| Callback-based history collection | Explicit `initialise`/`step` loop or CSMC history storage |
| Persistent RB reference beliefs | Outer-only `ReferenceTrajectory`; recompute inner beliefs each sweep |

Conditional SMC requires `BF(N; resampler=Multinomial())` (or a guided filter with the
same resampler). Systematic/stratified
conditional resampling is rejected: pinning one unconditional draw does not implement their
conditional laws. These resamplers remain available for ordinary particle filtering.

`AuxiliaryParticleFilter` supports conditional SMC with `NoRefreshment()` only. Its
ancestor-sampling and backward-simulation integrations are not implemented and reject
those combinations explicitly. Ancestor sampling on ordinary PF/RBPF resamples every
step, independently of the configured ESS threshold.

RB ancestor sampling and backward simulation require an analytical backward predictor.
They reject Kalman covariance repair, because the backward formulas describe the unrepaired
model. Nonzero backward-predictor jitter is also rejected by exact RB backward methods.
`NoRepair()` is the default. A repair also changes the computed likelihood and can
introduce nondifferentiable thresholds; it is not a substitute for a valid covariance model.

The default Gaussian backward predictor also factorises the process noise, observation
noise and predictive state covariances, so those matrices must be positive definite.
The forward Kalman likelihood can handle some singular inner covariances when its
innovation covariance remains positive definite. Such models can use `NoRefreshment()`
or provide a suitable custom analytical backward predictor.

GPU resampling remains optional and needs CUDA hardware for execution. CPU release tests do
not establish GPU runtime correctness. Automatic activity probing and reusable trajectory/AD
preparation are deferred until benchmark evidence motivates them.

With this DynamicPPL version, Julia 1.12 source loading with `--compiled-modules=no`
triggers an upstream generated-function binding error. Use Julia's normal precompiled
package loading. The ordinary loading path is verified separately from that source-only
failure; GeneralisedFilters does not override DynamicPPL internals to suppress it.
