# GeneralisedFilters 0.5 / SSMProblems 0.7 release review

This release replaces the 0.4.2 model interface. The consolidated development branch is
`th/closure-interface`; conditional-model and substrate changes are already incorporated.
The historical fast-diff branch is not a dependency of this release.

## Interface decisions

| Boundary | Release contract | Regression coverage |
|:--|:--|:--|
| Shared model interface | SSMProblems owns process abstracts, distribution adapters, the abstract model/accessor interface and standard container; GeneralisedFilters adds analytical components and conditioning | `test/models/substrate.jl`, `model_interface.jl` |
| Custom model containers | Accessors supply components to ordinary inference; conditional likelihoods, CSMC and Turing use shallow standard-container views preserving those components | `test/models/model_interface.jl`, `test/integrations/conditional_ad.jl` |
| Conditional inner model | Prior uses `x0`, transition uses adjacent outer states, observation uses current outer state; borrowed trajectories require rebuilding after changes | `test/models/conditional.jl` |
| Parameter target | Outer trajectory density plus conditional inner likelihood; parameter prior and Jacobian included once by the host | `test/integrations/logdensity.jl`, `turing.jl` |
| Gibbs alternation | Outer-only references; recompute inner beliefs and refresh HMC target after every trajectory change | `test/integrations/particle_gibbs.jl`, `apf_srkf_turing.jl` |
| Gaussian storage | Preserve structured model parameters; computational KF states use full dense/static covariance storage | `test/algorithms/kalman_storage.jl` |
| Differentiation | ForwardDiff through the primal; Mooncake static numerical rule and derived rules for other supported representations | `test/integrations/conditional_ad.jl`, `structured_covariance_ad.jl` |
| Particle weight types | Explicit zero markers, `add_logweight`, first-step promotion, stable later weights and preserved CSMC history precision | `test/components/weight_types.jl`, `weight_history.jl`, `test/integrations/particle_weight_ad.jl` |
| Particle histories | Public distribution-based constructors and checked append operations, separate initial-state types, shallow buffer ownership without implicit deep copies | `test/components/history_api.jl`, `weight_history.jl` |
| Conditional resampling | Conditional offspring law, including systematic/stratified; AS occurs at population resampling events | `test/components/resamplers.jl`, `test/algorithms/csmc.jl` |
| APF refreshment | Corrected filtering weights and selected-ancestor lookahead correction | `test/algorithms/apf_refreshment.jl` |
| Stable RB refreshment | Square-root backward messages; exact AS/BS rejects filtering-state repair and backward jitter | `test/algorithms/sqrt_backward.jl` |

These tests support the implemented contracts; they do not prove posterior correctness for
arbitrary user-defined processes or arbitrary numerical regularisation policies.

## Relationship to upstream fixes

[PR #189](https://github.com/TuringLang/SSMProblems.jl/pull/189) repairs storage and
symmetric covariance tangents in the old distribution-backed state implementation.
The new canonical computational state and structured-parameter regressions address the
corresponding defects without restoring that implementation.

[PR #190](https://github.com/TuringLang/SSMProblems.jl/pull/190), stacked on #189, addresses
accumulation, observation sensitivities, input representations, jitter, empty sequences and
precision. The new tests exercise repeated objectives, shared covariance/mean parameters,
existing gradient contributions, cache reuse, views, static time containers, selected
triangles of Symmetric/real Hermitian matrices, rejection of empty Kalman likelihood
inputs, and mixed precision.
They compare Mooncake with ForwardDiff; the structured covariance cases additionally
compare with finite differences.

The old `kf_loglikelihood` API and its PDMats-specific tangent plumbing are removed.
Tests concerning the private pairing of PDMats matrices and cached Cholesky factors do not
map to the new filtering-state contract. This is not a promise that every third-party
AbstractMatrix subtype has an AD implementation. Existing distribution interoperation
continues to be tested separately.

There is a deliberate jitter difference: #190 gives jitter a zero derivative, while the new
interface differentiates parameter-dependent repair. Repair of filtering states remains
incompatible with exact analytical RB AS/BS. Regularisation in model atoms must be shared
by every target evaluation.

Do not merge either old implementation wholesale. Before closing the PRs and issue #188,
link the consolidated replacement and its regression coverage; maintainers may separately
choose to retain fixes for a legacy release.

## Deferred work

- Lazy contractions for structured covariance derivatives. Correct derivatives currently
  use existing AD; the proposed optimisation is described in `structured-covariance-ad.md`.
- Persistent trajectory/AD workspaces and automatic activity probing. Preparation is
  rebuilt when the conditional target changes.
- An opt-in numerical-failure-as-zero-likelihood policy. Exceptions currently propagate;
  there is no automatic rejection of a failed CSMC sweep.
- Ancestor refreshment at skipped population-resampling steps and MH-corrected approximate
  backward proposals. Neither is required by the implemented refreshment schedule.

Turing supports one fixed-dimensional continuous `SSMTrajectory` variable, used through
our outer `ParticleGibbs` sampler. It does not provide a `Turing.Gibbs` component or arbitrary
analytic marginalisation of a Turing program. These limits are documented in the inference
guide and are not claims of future compatibility.

## Release gates

1. Run both packages' tests, AD/Turing regressions, strict documentation builds and formatter.
   GeneralisedFilters CI covers its minimum Julia 1.12.7 on Linux and current stable Julia
   on Linux, macOS and Windows. SSMProblems also tests Julia LTS.
2. Review the consolidated diff and obtain passing PR CI. Local Linux results cannot
   establish cross-platform or current-stable-Julia results.
3. Reconcile the legacy PRs and the separate dependency-update PR #191. GeneralisedFilters
   permits Turing 0.47–0.49; test results must record the actual selected versions.
4. Merge the reviewed branch. Release/register SSMProblems 0.7 first, then resolve and
   release/register GeneralisedFilters 0.5 against the registered dependency.
5. Verify published docs and installation from the registry. The monorepo Aqua
   persistent-task check is skipped while it cannot resolve the unreleased sibling; the
   registered installation enables that check.

Release tags and registration are separate from preparing and pushing this branch.

## Numerical-method development notes

The following records possible future changes for maintainers. It does not describe
additional user-facing options in this release.

### Numerical failures

This release propagates numerical exceptions from likelihood evaluation. It does not yet
provide a policy that converts factorisation failures to `-Inf` or automatically rejects a
CSMC sweep. Use consistent model regularisation and the square-root filter where needed.
An opt-in policy treating numerical failures as zero likelihood is deferred; it will need
to cover the parameter objective and trajectory update consistently.

### Possible MH correction of approximate backward weights

A future fallback could use approximate backward scores to propose an ancestor and apply
an MH correction with the target scores of only the current and proposed ancestors. With
a fixed number of proposals per step, suffix evaluation would cost `O(T^2)` in total,
alongside `O(NT)` particle/proposal work. This is the MH-within-PGAS construction in
[Lindsten et al., §6.1](https://jmlr.org/papers/volume15/lindsten14a/lindsten14a.pdf).
It is not currently an implemented refreshment strategy.

One global MH correction after an arbitrary approximate particle sweep is not automatically
valid: it requires the reverse proposal law, or a proven reversible proposal kernel for an
evaluable surrogate target. All target evaluations, including HMC, must remain consistent.
An MH correction still requires a stable target likelihood evaluator.
