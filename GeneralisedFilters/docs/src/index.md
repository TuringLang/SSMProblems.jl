# GeneralisedFilters

GeneralisedFilters is a flexible, modular framework for state-space inference in Julia.
A state-space model describes a hidden state that evolves over time and produces noisy
observations. Examples include tracking an object's position from sensor readings and
estimating an economic trend from observed prices.

The package separates the model from the algorithm used to infer its states. You define
an initial distribution, transition dynamics and an observation model. You can then choose
an inference method suited to the model's structure:

- **Filtering** estimates the current state using observations available so far.
- **Smoothing** estimates past states using later observations as well.
- **Particle Gibbs** alternates updates of a state trajectory and the model parameters.

Analytical methods, such as the Kalman filter, integrate out states where the model permits
it. Particle methods approximate distributions with weighted samples. GeneralisedFilters
supports both, with a particular focus on combining them for Rao–Blackwellised inference.

## Rao–Blackwellised state-space models

Consider a model with two hidden components: an underlying trend and a changing volatility.
The full model is not linear and Gaussian. But if the volatility trajectory were known,
the trend could be described by a linear-Gaussian model and integrated out with a Kalman
filter.

A **Rao–Blackwellised particle filter** takes advantage of this conditional structure.
It samples only the volatility. Each particle carries a Gaussian distribution for the
trend, which is updated analytically as observations arrive. Integrating out the trend
reduces the dimension of the sampling problem and can improve accuracy for a given
number of particles.

The guides call the sampled component the **outer state** and the analytically integrated
component the **inner state**. These refer to the roles of the two components in inference.
You specify how the inner model depends on the outer state, then compose the filters:

```julia
using GeneralisedFilters

pf = RBPF(BF(100), KF())
```

Here `BF(100)` is a bootstrap particle filter with 100 particles, and `KF()` is a Kalman
filter for each particle's conditional inner model. The same model can be used for particle
Gibbs, which updates the outer trajectory while integrating out the inner states. Parameter
updates can use HMC or NUTS through the Turing.jl integration. ForwardDiff and Mooncake
provide forward and reverse mode differentiation of the conditional likelihood.

## A first filtering example

Install the package and load it:

```julia
import Pkg
Pkg.add("GeneralisedFilters")
using GeneralisedFilters
```

These pages describe the 0.5 interface. To try it from this repository before its release,
run the following from the repository root instead of `Pkg.add`:

```julia
using Pkg
Pkg.develop([PackageSpec(path="SSMProblems"), PackageSpec(path="GeneralisedFilters")])
```

Start with a simple linear-Gaussian model. Its scalar state decays toward zero with Gaussian
process noise. Each observation measures that state with additional Gaussian noise.
The prior describes time zero, and the three observations correspond to times 1, 2 and 3.
One-element vectors and matrices express the scalar model in the Kalman filter's array
interface.

```@example getting_started
using GeneralisedFilters
using Statistics

model = StateSpaceModel(
    GaussianPrior([0.0], fill(1.0, 1, 1)),
    LinearGaussianDynamics(fill(0.9, 1, 1), [0.0], fill(0.1, 1, 1)),
    LinearGaussianObservation(fill(1.0, 1, 1), [0.0], fill(0.2, 1, 1)),
)
observations = [[0.2], [-0.1], [0.3]]

state, loglikelihood = GeneralisedFilters.filter(model, KF(), observations)
(mean(state), cov(state), loglikelihood)
```

The result contains the Gaussian filtering distribution at time 3 and the log likelihood
of all three observations. Filtering requires at least one observation. The covariance arguments specify variances, not standard
deviations. To obtain only the likelihood, use
`marginal_loglikelihood(model, KF(), observations)`.

The model and `KF()` are separate objects. That separation also applies when you introduce
time-varying components, custom processes or a hierarchical model. Small, fixed-dimensional
models can use StaticArrays in place of ordinary arrays.

## Where to go next

Read [Models and conditioning](models/linear-gaussian.md) to define a hierarchical model
and see how fixing an outer trajectory gives a conditional inner model. Then follow
[Particle Gibbs and Turing](inference.md) for joint trajectory and parameter inference.
Read [Recording filtering results](history.md) for manual loops and particle ancestry
storage. The **Examples** section works through trend inflation with stochastic volatility.

The [API reference](api.md) lists the available types and operations. Existing users
upgrading from 0.4.2 can consult the [migration guide](migration.md).

The shared model interface is provided by SSMProblems and re-exported by GeneralisedFilters.
You do not need to import SSMProblems separately. Turing and the differentiation backends
are optional dependencies, needed when using their respective integrations.
