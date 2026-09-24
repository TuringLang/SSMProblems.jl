# Changelog

## 0.7.0

This release supplies the shared model interface for GeneralisedFilters 0.5.

- `StatePrior`, `LatentDynamics` and `ObservationProcess` are now abstract types
  without scalar/state type parameters. Concrete components retain their own type
  parameters and fields.
- `distribution`, `simulate` and `logdensity` no longer forward keyword arguments.
  Store parameters and controls in components and index time-varying inputs by `t`.
- Define distribution-based processes with `DistributionPrior`, `DistributionDynamics`
  and `DistributionObservation`. Dynamics and observation functions receive `(t, state)`,
  allowing time-varying parameters without a custom process type.
- `StateSpaceModel(prior, dyn, obs)` and forward simulation belong to this package;
  inference algorithms and conditioning remain downstream. Custom containers subtype
  `AbstractStateSpaceModel` and implement `prior`, `dyn` and `obs`. The constructor
  `StateSpaceModel(model)` retains the components returned by these accessors.
- Remove the AbstractMCMC dependency. `AbstractStateSpaceModel` remains available but
  no longer inherits from `AbstractMCMC.AbstractModel`. Replace `sample(rng, model, T)`
  with `simulate(rng, model, T)`. Component-level `simulate` methods require an explicit RNG.
- Forward simulation returns `(x0, xs, ys)`, supports zero-length trajectories and
  rejects negative lengths. Distribution-based simulation preserves static vectors
  for multivariate normal distributions with static means.

Release SSMProblems 0.7 before GeneralisedFilters 0.5. AdvancedPS versions requiring
SSMProblems 0.6 cannot share an environment with this release. GeneralisedFilters's
Turing integration supports Turing 0.47–0.49, which no longer depends on AdvancedPS.
