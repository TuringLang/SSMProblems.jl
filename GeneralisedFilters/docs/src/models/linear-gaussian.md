# Models and conditioning

A model consists of a prior, dynamics, and observations. Linear-Gaussian atoms retain
parameter arrays, including `Diagonal` and `Symmetric` covariance representations.
Kalman filtering states store full covariances in `GaussianState` without eager
factorisation: `SVector` means use `SMatrix` covariances, while dynamic means use ordinary
vectors and matrices. Conversion respects the selected triangle of `Symmetric` and promotes
the mean and covariance scalar types together, including ForwardDiff Dual values.
Model parameter objects are preserved. Covariances must be symmetric, and innovation
covariances must be positive definite for the Kalman likelihood.

The smoother establishes history storage after the first update, allowing initial scalar
promotion; subsequent states must retain that storage and scalar type. `marginal_loglikelihood`
accumulates in at least Float64 precision, retaining wider and AD scalar types. This gives
Float32/Float64 models a consistent scalar return type for empty and nonempty data without
forcing Float32 filtering states or individual likelihood increments to Float64.

Structured covariance gradients currently use ordinary AD through their parameterisation;
only the existing plain-static Kalman step uses the handwritten numerical rule. Direct
structured-gradient optimisations remain deferred.

```@example models
using GeneralisedFilters, StaticArrays
model = StateSpaceModel(
    GaussianPrior(SA[0.0], SMatrix{1,1}(1.0)),
    LinearGaussianDynamics(SMatrix{1,1}(0.9), SA[0.0], SMatrix{1,1}(0.1)),
    LinearGaussianObservation(SMatrix{1,1}(1.0), SA[0.0], SMatrix{1,1}(0.2)),
)
ys = [SA[0.2], SA[-0.1], SA[0.3]]
state, ll = GeneralisedFilters.filter(model, KF(), ys)
@assert ll ≈ marginal_loglikelihood(model, KF(), ys)
state
```

For time variation, use `TimeVaryingDynamics` or `TimeVaryingObservation` wrapping a function
of `(; t)`. Define custom nonlinear processes with `DistributionPrior`,
`DistributionDynamics((t, x) -> distribution)`, and
`DistributionObservation((t, x) -> distribution)`, or implement the local `simulate` and
`logdensity` methods on process subtypes.

## Conditional models

Define parameter dependence in an ordinary builder. Compute time-independent quantities
once inside it, and capture controls from outside it. A hierarchical inner prior resolves
with `(; x0)`, inner dynamics with `(; t, x_prev, x_new)`, and inner observations with
`(; t, x)`.

```@example models
using Distributions
function build(θ)
    q = exp(θ[1]) * SMatrix{1,1}(1.0)
    inner_dyn((; t, x_prev, x_new)) = LinearGaussianDynamics(
        SMatrix{1,1}(0.8), SA[0.1x_prev + 0.2x_new], q,
    )
    return StateSpaceModel(
        DistributionPrior(Normal()),
        DistributionDynamics((t, x) -> Normal(0.9x, 0.3)),
        GaussianPrior(SA[0.0], SMatrix{1,1}(1.0)),
        inner_dyn,
        LinearGaussianObservation(SMatrix{1,1}(1.0), SA[0.0], SMatrix{1,1}(0.2)),
    )
end
outer = ReferenceTrajectory(0.1, [0.2, -0.3, 0.4])
hierarchical = build([-1.0])
inner = condition_inner(hierarchical, outer)
@assert inner_loglikelihood(KF(), hierarchical, outer, ys) ≈
    marginal_loglikelihood(inner, KF(), ys)
trajectory_logdensity(hierarchical, KF(), outer, ys)
```

`condition_inner` resolves the prior once and returns a lightweight ordinary SSM with lazy
transition/observation components. It borrows the trajectory: rebuild it after changing the
trajectory or parameters. An ordinary vector stores `x0` at index 1; `ReferenceTrajectory`
uses indices `0:T`. There must be exactly one more state than observations.

The four-argument `trajectory_logdensity` adds the outer initial and transition densities
to the inner marginal likelihood. It does not add a parameter prior or transformation
Jacobian. Conditional-model construction does not itself supply backward-sampling or
ancestor-sampling capabilities for a new analytical filter.

For a scalar objective, differentiate
`θ -> trajectory_logdensity(build(θ), KF(), outer, ys)` with the desired backend.
No activity annotations are required. `with_activity` is an optional optimisation whose
manual flags must be checked against finite differences.
