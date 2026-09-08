# Changelog

## 0.6.0

This release changes the model interface and requires migration from 0.5. It targets
Julia 1.12.7 and the current Turing 0.46 / DynamicPPL 0.42 integration.

- Define models with Gaussian/discrete atoms and conditioning closures. GeneralisedFilters
  owns the process interface; SSMProblems and PDMats are no longer runtime dependencies.
- Use `condition_inner` to expose a hierarchical model's conditional inner SSM. One
  analytical likelihood evaluator serves both ordinary and conditional models.
- Share inner component resolution between simulation, densities and Rao–Blackwellised
  particle filtering. Proposals receive the full particle state.
- Store outer-only conditional-SMC references and recompute inner filtering states.
- Refresh cached parameter-sampler targets when trajectories change. Turing uses the same
  marginalised trajectory objective and handles constrained parameters.
- Support ForwardDiff and a Mooncake reverse rule for static-array Kalman likelihoods.
  Observation sensitivities and eigenvalue-clipping sensitivities are propagated.
- Reject unsupported conditional systematic/stratified resampling and repaired-Kalman
  backward sampling rather than apply mathematically incompatible backward weights.
- Stabilise finite-state filtering and smoothing at extreme likelihoods and unreachable states.
- Correct CUDA resampling sample counts, RNG handling and stored trajectory element types.
- Replace callbacks with explicit filtering loops or CSMC history storage.

Automatic activity probing and persistent AD/trajectory workspaces remain deferred.
