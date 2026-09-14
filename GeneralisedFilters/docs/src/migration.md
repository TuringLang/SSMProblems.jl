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

Conditional SMC requires a resampler that implements a conditional law:
`Multinomial()`, `Systematic()` and `Stratified()` all do, following Finke, Johansen, Lee
& Murray, "Resampling in conditional SMC algorithms" (arXiv:2606.25603). `Metropolis()`
and `Rejection()` are rejected, because that reference derives no index distribution for
them; they remain available for ordinary particle filtering. Custom resamplers opt in by
implementing `conditional_sample_ancestors` and `supports_conditional`.

Conditioning is no longer applied by overwriting one index of an unconditional draw. That
shortcut works for independent multinomial draws; it is not a valid general
conditional law for dependent schemes.

`AuxiliaryParticleFilter` supports `NoRefreshment()`, `AncestorSampling()` and
`BackwardSimulation()`, including wrapped RBPFs. Lookahead weights affect ordinary ancestor
selection; backward weights use the corrected filtering weights. A selected reference
ancestor receives its own inverse-lookahead correction.

Ancestor sampling now respects the ESS threshold. The implemented schedule refreshes the
reference ancestor **at resampling events**; when population resampling is skipped, it keeps
ancestors and accumulated filtering weights. This does not implement every-step ancestor
refreshment alongside skipped population resampling. Conditional systematic/stratified
resampling draws the entire offspring law conditional on the selected reference ancestor.

RB ancestor sampling and backward simulation require an analytical backward predictor.
`KF()` and `SRKF()` now default to `SqrtBackwardInformationPredictor()`, which represents
backward likelihoods as factored Gaussian residuals and uses QR and triangular solves.
The explicit legacy `BackwardInformationPredictor()` remains available; its scalar
precision-cancellation defect is fixed, but the factored predictor is recommended for
ill-conditioned models.

Use `SRKF()` for square-root forward filtering as well. `CovarianceFactor(F)` represents
`F * F'` without adding noise; supplied prior and process factors may be rectangular or
rank deficient. Ordinary covariance matrices still require Cholesky factorization on this
route. Observation noise must be positive definite. A singular Gaussian does not acquire
a full-dimensional `logpdf` merely because its filtering factors are supported.

Exact RB AS/BS continue to reject filtering-state covariance repair and nonzero backward
jitter: their analytical messages must match the forward likelihood. If regularization is
needed, define it explicitly in the model's covariance atoms, consistently for filtering,
refreshment, simulation and HMC. Square-root evaluation changes the numerical representation,
not the statistical target. Adaptive clipping of a filtered covariance is a different
operation and generally does not admit the same shared backward message.

GPU resampling remains optional and needs CUDA hardware for execution. CPU release tests do
not establish GPU runtime correctness. Automatic activity probing and reusable trajectory/AD
preparation are deferred until benchmark evidence motivates them.

With this DynamicPPL version, Julia 1.12 source loading with `--compiled-modules=no`
triggers an upstream generated-function binding error. Use Julia's normal precompiled
package loading. The ordinary loading path is verified separately from that source-only
failure; GeneralisedFilters does not override DynamicPPL internals to suppress it.
