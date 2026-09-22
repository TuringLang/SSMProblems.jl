# Models and conditioning

A state-space model describes an unobserved process and the measurements we make of it.
For example, a sensor might record noisy measurements of a signal that changes over time.
Inference uses those measurements to estimate the signal and, when needed, learn the
parameters that govern its evolution.

GeneralisedFilters separates a model into three components:

- A **prior** describes the initial state at time zero.
- **Dynamics** describe how the state changes between consecutive times.
- An **observation process** describes a measurement given the current state.

You combine these components in a `StateSpaceModel` and choose an inference algorithm
separately. This lets you reuse a model with different algorithms, or change one component
without rewriting the rest of the model.

## A linear-Gaussian model

Consider a scalar signal ``z_t`` observed with Gaussian noise:

```math
z_0 \sim \mathcal{N}(0, 1), \qquad
z_t = 0.9 z_{t-1} + w_t, \qquad
y_t = z_t + v_t,
```

where ``w_t \sim \mathcal{N}(0, 0.1)`` and ``v_t \sim \mathcal{N}(0, 0.2)`` are independent.
The second arguments here are variances. The corresponding model uses three Gaussian
components. Their vector and matrix arguments also support multivariate states.

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

`KF()` selects the Kalman filter. It returns the Gaussian distribution of the final state
given all observations, along with the log marginal likelihood of the observations. Use
`marginal_loglikelihood` when you only need that scalar likelihood.

This example uses StaticArrays for its small, fixed-size state. Ordinary Julia vectors and
matrices work too. Observations start at time one: the filter first propagates the initial
state, then conditions on the first observation.

## Rao-Blackwellised models

Some models become linear and Gaussian only after conditioning on another latent process.
For example, suppose a signal's process variance changes over time. Let ``x_t`` describe
its log variance and ``z_t`` the signal:

```math
\begin{aligned}
x_0 &\sim \mathcal{N}(0, 1), &
x_t \mid x_{t-1} &\sim \mathcal{N}(0.9x_{t-1}, q), \\
z_0 &\sim \mathcal{N}(0, 1), &
z_t \mid z_{t-1}, x_t &\sim \mathcal{N}(0.8z_{t-1}, \exp(x_t)), \\
&& y_t \mid z_t &\sim \mathcal{N}(z_t, 0.2).
\end{aligned}
```

The joint model is not linear-Gaussian. However, if we fix the log-variance trajectory
``x_{0:T}``, a Kalman filter can integrate out the signal ``z_{0:T}`` exactly. A
Rao-Blackwellised particle filter uses this structure: it samples trajectories of ``x``
and keeps a Gaussian filtering distribution for ``z`` within each particle. This reduces
the part of the state space that must be represented by samples.

The interface calls ``x`` the **outer state** and ``z`` the **inner state**. The
five-component model constructor takes the outer prior and dynamics, followed by the inner
prior, dynamics, and observation process. Inner components can depend on the outer state
through functions that return the appropriate model component.

```@example models
using Distributions
function build(q)
    inner_dyn((; t, x_prev, x_new)) = LinearGaussianDynamics(
        SMatrix{1,1}(0.8), SA[0.0], SMatrix{1,1}(exp(x_new)),
    )
    return StateSpaceModel(
        DistributionPrior(Normal()),
        DistributionDynamics((t, x) -> Normal(0.9x, sqrt(q))),
        GaussianPrior(SA[0.0], SMatrix{1,1}(1.0)),
        inner_dyn,
        LinearGaussianObservation(SMatrix{1,1}(1.0), SA[0.0], SMatrix{1,1}(0.2)),
    )
end
hierarchical = build(0.1)
```

Combine a bootstrap particle filter for the outer states with a Kalman filter for the inner
states to filter this model:

```@example models
using Random
rng = Xoshiro(42)
particles, loglikelihood_estimate = GeneralisedFilters.filter(
    rng, hierarchical, RBPF(BF(100), KF()), ys,
)
loglikelihood_estimate
```

Here `q` is a model parameter, and `inner_dyn` uses the current outer state to set the
signal's process variance. A builder such as `build` is an ordinary Julia function. It can
compute quantities shared across time once and capture data or controls from its enclosing
scope.

Each inner component receives a named tuple with the conditioning values available to it:

| Component | Input to its function | Returned component |
| :--- | :--- | :--- |
| Initial prior | `(; x0)` | A prior for the inner initial state |
| Dynamics | `(; t, x_prev, x_new)` | Inner dynamics for this transition |
| Observations | `(; t, x)` | Inner observation process at this time |

A component that does not depend on these values can be supplied directly, as the inner
prior and observation process are above. If dynamics need a longer outer history, include
that history in the outer state so each transition has the information it needs.

## Conditioning on a trajectory

`condition_inner` turns a hierarchical model and a fixed outer trajectory into an ordinary
state-space model. You can then apply the same Kalman likelihood calculation used above.

```@example models
outer = ReferenceTrajectory(0.1, [0.2, -0.3, 0.4])
inner = condition_inner(hierarchical, outer)
@assert inner_loglikelihood(KF(), hierarchical, outer, ys) ≈
    marginal_loglikelihood(inner, KF(), ys)
trajectory_logdensity(hierarchical, KF(), outer, ys)
```

`ReferenceTrajectory` stores the initial state separately and uses time indices `0:T`.
You may also pass an ordinary vector `[x0, x1, ..., xT]`. Either representation must contain
one more state than there are observations.

The two density evaluations serve different purposes:

- `inner_loglikelihood` evaluates ``\log p(y_{1:T} \mid x_{0:T})``, integrating out the
  inner states.
- `trajectory_logdensity` also adds the outer initial and transition densities, giving
  ``\log p(x_{0:T}, y_{1:T})``. It does not include a prior on model parameters or a
  parameter-transformation Jacobian.

For parameter inference, differentiate an objective such as
`θ -> trajectory_logdensity(build(exp(θ[1])), KF(), outer, ys)`. The outer trajectory stays
fixed while parameter changes affect the model. See [Particle Gibbs and Turing](../inference.md)
for a sampler that alternates parameter and trajectory updates.

The conditional model borrows its trajectory. Keep that trajectory unchanged while using
the model, and rebuild the conditional model when the trajectory or parameters change.
The prior is resolved at construction, while dynamics and observations are resolved as the
filter visits each time step.

## Other model components

For time-varying linear-Gaussian models, wrap a function of `(; t)` in
`TimeVaryingDynamics` or `TimeVaryingObservation`. The function returns the component for
that time step.

For nonlinear or non-Gaussian components, use `DistributionPrior`,
`DistributionDynamics((t, x) -> distribution)`, and
`DistributionObservation((t, x) -> distribution)`. You can also define your own process
subtypes and implement their `simulate` and `logdensity` methods. These constructors make
a model usable by compatible algorithms. Analytical filtering, ancestor sampling, and
backward simulation each require the corresponding algorithm support for its components.

## Covariance storage and differentiation

Gaussian model components retain the arrays you supply, including `Diagonal`, `Symmetric`,
and real `Hermitian` covariances. During Kalman filtering, the state distribution uses a
full covariance because prediction and conditioning can introduce correlations. `SVector`
means use `SMatrix` covariances. Dynamic means use ordinary vectors and matrices. This
conversion respects the selected triangle of symmetric wrappers and promotes mean and
covariance scalar types together, including ForwardDiff dual numbers.

Automatic differentiation follows the parameterisation of structured model covariances.
Mooncake also provides a specialised Kalman derivative rule for plain static arrays.
Activity annotations are not required. If you use `with_activity` to mark inactive inputs
manually, check those declarations against finite differences.

Covariances must be symmetric, and innovation covariances must be positive definite for
the Kalman likelihood. For numerical stability and square-root filtering, see
[Numerical stability](../inference.md#Numerical-stability).

Two type conventions matter when writing generic numerical code. The smoother allocates
its history after the first update, so later states must retain that storage and scalar
type. Also, `marginal_loglikelihood` accumulates in at least Float64 precision while
preserving wider and AD scalar types. Float32 filtering states and individual likelihood
increments can remain Float32.
