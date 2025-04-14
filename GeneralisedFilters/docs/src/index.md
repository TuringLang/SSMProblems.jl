# GeneralisedFilters

## Installation

In the `julia` REPL:

```julia
] add GeneralisedFilters
```

## Documentation

`GeneralisedFilters` provides implementations of various filtering and
smoothing algorithms for state-space models (SSMs). The goal of the package is
to provide a modular and extensible framework for implementing advanced
algorithms including Rao-Blackwellised particle filters, two-filter smoothers,
and particle Gibbs/conditional SMC. Performance is a primary focus of this work,
with type stability, GPU-acceleration, and efficient history storage being key
design goals.

### Interface
```@autodocs
Modules = [SSMProblems]
Order   = [:type, :function, :module]
```
