# GeneralisedFilters.jl

GeneralisedFilters is a flexible, modular framework for state-space inference in Julia.
State-space models describe an unobserved process that evolves over time and is measured
through noisy observations. The package provides filtering to estimate the current state,
smoothing to infer past states, and particle Gibbs to jointly infer state trajectories and
model parameters.

Models and inference algorithms are defined separately. You can combine prior, transition
and observation models, then choose an analytical filter, a particle filter, or a combination
of the two. Custom processes and algorithms can extend the same interface.

## Rao–Blackwellised inference

Many models have a part that is difficult to integrate out and another part that becomes
linear and Gaussian once the first is known. For example, a model of inflation might have
an unknown trend and changing volatility. Given the volatility trajectory, a Kalman filter
can integrate out the trend.

A Rao–Blackwellised particle filter uses this structure: particles sample the volatility,
while each particle carries a conditional Gaussian distribution for the trend. This reduces
the number of states that must be sampled and can give more accurate estimates for a given
number of particles.

This combination is a particular focus of GeneralisedFilters. A particle filter and an
analytical filter can be composed directly:

```julia
using GeneralisedFilters

pf = RBPF(BF(100), KF())  # 100 outer particles, each with an inner Kalman filter
```

For joint state and parameter inference, particle Gibbs alternates trajectory updates with
parameter updates. The Turing.jl integration lets you specify parameter priors in a Turing
model and use HMC or NUTS for the parameter update. ForwardDiff and Mooncake provide forward
and reverse mode differentiation of the likelihood after integrating out the Gaussian states.
StaticArrays are supported for small, fixed-dimensional states.

## Getting started

The [documentation overview](GeneralisedFilters/docs/src/index.md) introduces the model
and algorithm interface with a complete filtering example. From there:

- [Models and conditioning](GeneralisedFilters/docs/src/models/linear-gaussian.md) explains
  how to define models and their Rao–Blackwellised structure.
- [Particle Gibbs and Turing](GeneralisedFilters/docs/src/inference.md) covers joint inference
  for trajectories and parameters.
- [Recording filtering results](GeneralisedFilters/docs/src/history.md) shows manual
  loops and particle ancestry storage.
- The [trend inflation example](GeneralisedFilters/examples/trend-inflation/script.jl)
  applies Rao–Blackwellised filtering to a model with stochastic volatility.
- The [static arrays example](GeneralisedFilters/examples/static-arrays/script.jl)
  benchmarks the speed-up from using StaticArrays for small states.

This repository contains both GeneralisedFilters and SSMProblems, which supplies the shared
state-space model interface. GeneralisedFilters re-exports that interface, so most users
only need to load GeneralisedFilters.
