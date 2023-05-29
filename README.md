# SSMProblems.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://turinglang.github.io/SSMProblems.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://turinglang.github.io/SSMProblems.jl/dev)
[![Build Status](https://github.com/TuringLang/AdvancedPS.jl/workflows/CI/badge.svg?branch=master)](https://github.com/TuringLang/SSMProblems.jl/actions?query=workflow%3ACI%20branch%3Amaster)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

A minimalist framework to define State Space Models (SSM) and their associated logdensities to feed into inference algorithms.

### Basic interface
This package defines the basic interface needed to run inference on State Space Models as the following:
```julia
# State wrapper
abstract type AbstractParticle end

"""
Emits a new state candidate from latent dynamics
"""
function transition!! end

"""
Scores the emission transition
"""
function emission_logdensity end

"""
Stops the state machine
"""
function isdone end

```

### Linear Gaussian State Space Model
As a concrete example, the following snippet of pseudo-code defines a linear gaussian state space model:
```julia
using SSMProblems
using Distributions

# Model definition
T = 10
sig_u = 0.1
sig_v = 0.2
observations = ...

struct LinearSSM{T} <: AbstractParticle
    state::T
end

function transition!!(rng, step, particle::LinearSSM)
    if step == 1
        return rand(rng, Normal())
    end
    return rand(rng, Normal(particle.state, sig_u))
end

function emission_logdensity(step, particle::LinearSSM)
    return logpdf(Normal(particle.state, sig_v), observations[step])
end

isdone(step, ::LinearSSM) = step > T
```

More details can be found in the [documentation]() and the [examples]().
