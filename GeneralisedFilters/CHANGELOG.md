# Changelog

## 0.6.0 (unreleased)

- **Compatibility:** Kalman initialisation selects full covariance storage; subsequent
  updates and smoothing retain the array representations and scalar types produced by
  their arithmetic. Mean and covariance are no longer promoted together automatically.
  Models should keep compatible state storage across steps rather than rely on repeated
  canonicalisation of changing matrix representations.
- Add GPU Rao–Blackwellised particle filtering through the optional BatchedKernels 0.3
  extension, sharing numerical kernels and population orchestration with CPU filtering.
  Select device initialisation with `GPUExecution()` and provide CPU and batched methods
  on one model.
- Support GPU reference trajectories, conditional resampling, ancestor sampling, backward
  simulation and conditional Gaussian smoothing. Selected-path backward messages and
  particle-Gibbs parameter updates use CPU model methods; population scoring stays on GPU.
- Support sparse GPU particle histories with nested batched states and compact selected
  trajectories. Backward simulation retains dense history for past candidate populations.
- Add `CombinedRNG` to pass CPU and GPU random streams through one filtering or sampling
  interface. Runtime shared floating-point scalars use BK 0.3 without specialising kernels
  on their values.

- Add `ThreadedExecution` for evaluating particle populations on several threads, selected
  with the `execution` keyword of `ParticleFilter` and `BF`. RBPF, auxiliary particle
  filters and CSMC ancestor sampling and backward simulation use the setting of their
  particle filter. Results depend on the generator state and `blocksize` but not on the
  number of threads. `SerialExecution()` remains the default and is unchanged.

## 0.5.0

This release changes the model interface and requires migration from 0.4.2. It targets
Julia 1.12.7 and the Turing 0.47–0.49 / DynamicPPL 0.42 integration.

- Define models with Gaussian/discrete atoms and conditioning closures. GeneralisedFilters
  builds on the shared process interface in SSMProblems 0.7; PDMats is no longer a runtime dependency.
- Re-export SSMProblems' distribution adapters and abstract model/accessor interface.
  Custom model containers work with filtering, smoothing, conditional likelihoods, CSMC
  and `SSMTrajectory`. Hierarchical dynamics can depend on time and adjacent outer states.
  Component factories report invalid return types with guidance on the required interface.
- Use `condition_inner` to expose a hierarchical model's conditional inner SSM. One
  analytical likelihood evaluator serves both ordinary and conditional models.
- Share inner component resolution between simulation, densities and Rao–Blackwellised
  particle filtering. Proposals receive the full particle state.
- Require nonempty observations in `filter` as well as the Kalman likelihood, avoiding an
  empty-data return that changes the inferred likelihood type.
- Infer particle-weight scalar types from density contributions. Use `add_logweight` in
  custom particle updates, preserve CSMC history precision, and require stable weight types
  after the first completed step. Initial zero markers no longer act as generic numbers.
- Store outer-only conditional-SMC references and recompute inner filtering states.
- Refresh cached parameter-sampler targets when trajectories change. Turing uses the same
  marginalised trajectory objective and handles constrained parameters.
- Support ForwardDiff and a Mooncake reverse rule for static-array Kalman likelihoods.
  Observation sensitivities and eigenvalue-clipping sensitivities are propagated.
- Normalise Kalman state storage independently of structured covariance parameters; fix
  smoother histories after scalar promotion and reverse AD for shared structured covariances.
  Kalman marginal likelihoods require nonempty observations and initialise their total
  from the first increment, preserving its natural scalar type.
- Implement conditional multinomial/systematic/stratified resampling, conditioning the
  entire offspring law on the chosen ancestor. AS respects ESS and refreshes ancestors
  at resampling events.
- Support auxiliary particle filters with ancestor sampling and backward simulation,
  including Turing parameter updates for wrapped RBPFs.
- Use QR-based square-root Gaussian backward messages by default; fix catastrophic
  cancellation in the legacy information predictor. Explicit covariance factors support
  rank-deficient prior/process noise with SRKF.
- Reject filtering-state covariance repair with analytical backward sampling; explicit
  model regularization must be shared by forward, backward and parameter updates.
- Stabilise finite-state filtering and smoothing at extreme likelihoods and unreachable states.
- Correct CUDA resampling sample counts, RNG handling and stored trajectory element types.
  Support CUDA 5 and 6, including bulk-only device RNGs; run GPU regressions in Buildkite.
- Replace callbacks with explicit filtering loops and public particle-container operations.
  Constructors infer state types from the first result when requested, checked `push!`
  operations preserve initial ancestry, and containers copy buffers without deep-copying states.

Automatic activity probing and persistent AD/trajectory workspaces remain deferred.
