# SSMProblems.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](turinglang.org/SSMProblems.jl/SSMProblems/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](turinglang.org/SSMProblems.jl/SSMProblems/dev/)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
<!--[![Build Status](https://github.com/TuringLang/SSMProblems.jl/workflows/CI/badge.svg?branch=master)](https://github.com/TuringLang/SSMProblems.jl/actions?query=workflow%3ACI%20branch%3Amaster) -->


A minimalist framework to define state space models (SSMs) and their associated
log-densities to feed into inference algorithms.

## Basic interface

This package defines the basic interface needed to run inference on state space
as the following:

```julia
# Wrapper for model dynamics and observation process
abstract type LatentDynamics end
abstract type ObservationDynamics end

# Define the initialisation/transition distribution for the latent dynamics
function distribution(dyn::LatentDynamics, ...) end

# Define the observation distribution
function distribution(obs::ObservationProcess, ...) end

# Combine the latent dynamics and observation process to form a SSM
model = StateSpaceModel(dyn, obs)
```

For specific details on the interface, please refer to the package [documentation](https://turinglang.github.io/SSMProblems.jl/dev).

## Linear Gaussian State Space Model

As a concrete example, the following snippet of pseudo-code defines a linear
Gaussian state space model. Note the inclusion of the `extra` parameter in each
method definition. This is a key feature of the SSMProblems interface which
allows for the definition of more complex models in a performant fashion,
explained in more details in the package documentation.

```julia
using SSMProblems, Distributions

# Model parameters
sig_u, sig_v  = 0.1, 0.2

struct LinearGaussianLatentDynamics <: LatentDynamics end

# Initialisation distribution
function distribution(dyn::LinearGaussianLatentDynamics, extra::Nothing)
    return Normal(0.0, sig_u)
end

# Transition distribution
function distribution(
    dyn::LinearGaussianLatentDynamics,
    step::Int,
    state::Float64,
    extra::Nothing
)
    return Normal(state, sig_u)
end

struct LinearGaussianObservationProcess <: ObservationProcess end

# Observation distribution
function distribution(
    obs::LinearGaussianObservationProcess,
    step::Int,
    state::Float64,
    extra::Nothing
)
    return Normal(state, sig_v)
end
```
