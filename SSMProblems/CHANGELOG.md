# Changelog

## 0.7.0

This release supplies the shared model interface for GeneralisedFilters 0.6.

- `StatePrior`, `LatentDynamics` and `ObservationProcess` are now abstract types
  without scalar/state type parameters. Concrete components retain their own type
  parameters and fields.
- `distribution`, `simulate` and `logdensity` no longer forward keyword arguments.
  Store parameters and controls in components and index time-varying inputs by `t`.
- `StateSpaceModel(prior, dyn, obs)` and forward simulation belong to this package;
  inference algorithms and conditioning remain downstream. The dependency on
  AbstractMCMC is removed. Replace `sample(rng, model, T)` with
  `simulate(rng, model, T)`. `AbstractStateSpaceModel` is removed; the container is
  no longer an `AbstractMCMC.AbstractModel`. Access components through `model.prior`,
  `model.dyn` and `model.obs` instead of accessor functions. Component-level
  `simulate` methods now require an explicit RNG.
- Forward simulation returns `(x0, xs, ys)`, supports zero-length trajectories and
  rejects negative lengths. Distribution-based simulation preserves static vectors
  for multivariate normal distributions with static means.

Release SSMProblems 0.7 before GeneralisedFilters 0.6. AdvancedPS versions requiring
SSMProblems 0.6 cannot share an environment with this release. GeneralisedFilters's
Turing integration requires Turing 0.47 or 0.48, which no longer depends on AdvancedPS.
