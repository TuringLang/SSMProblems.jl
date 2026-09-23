# Changelog

## 0.5.0

This release changes the model interface and requires migration from 0.4.2. It targets
Julia 1.12.7 and the Turing 0.47–0.49 / DynamicPPL 0.42 integration.

- Define models with Gaussian/discrete atoms and conditioning closures. GeneralisedFilters
  builds on the shared process interface in SSMProblems 0.7; PDMats is no longer a runtime dependency.
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
- Replace callbacks with explicit filtering loops or CSMC history storage.

Automatic activity probing and persistent AD/trajectory workspaces remain deferred.
