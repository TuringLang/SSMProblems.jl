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
abstract type AbstractStateSpaceModel end

"""
Emits a new state candidate from latent dynamics
"""
function transition!! end

"""
Scores the emission transition
"""
function emission_logdensity end


```

### Linear Gaussian State Space Model
As a concrete example, the following snippet of pseudo-code defines a linear Gaussian state space model:
```julia
using SSMProblems, Distributions, Random

# Model definition
T, sig_u, sig_v  = 10, 0.1, 0.2
observations = rand(T)

struct LinearSSM <: AbstractStateSpaceModel end

# Model dynamics
function transition!!(rng::AbstractRNG, model::LinearSSM)
    return rand(rng, Normal(0, 1))
end

function transition!!(rng::AbstractRNG, model::LinearSSM, state::Float64, ::Int)
    return rand(rng, Normal(state, 1))
end

function emission_logdensity(model::LinearSSM, state::Float64, observation::Float64, ::Int)
    return logpdf(Normal(0, 1), observation)
end
```
