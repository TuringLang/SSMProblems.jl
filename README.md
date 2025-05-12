# SSMProblems.jl

[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
<!--[![Build Status](https://github.com/TuringLang/SSMProblems.jl/workflows/CI/badge.svg?branch=master)](https://github.com/TuringLang/SSMProblems.jl/actions?query=workflow%3ACI%20branch%3Amaster) -->

|           Package            |                                                                                                                                                    Docs                                                                                                                                                    |
| :--------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   SSMProblems   |   [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://turinglang.org/SSMProblems.jl/SSMProblems/dev/)        |
| GeneralisedFilters | [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://turinglang.org/SSMProblems.jl/GeneralisedFilters/dev/)|

A minimalist framework to define state space models (SSMs) and their associated
log-densities to feed into inference algorithms.


## Talk at [LAFI 2025](https://popl25.sigplan.org/details/lafi-2025/11/State-Space-Model-Programming-in-Turing-jl)

[PDF Slides](https://github.com/user-attachments/files/20160397/LAFI_2025_Presentation.pdf)

[![State space programming](http://i3.ytimg.com/vi/58DsScclqGU/hqdefault.jpg)](https://www.youtube.com/watch?v=58DsScclqGU
)


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
