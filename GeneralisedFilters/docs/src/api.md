# API reference

Use this page to look up constructors and operations. For an introduction with examples,
start with [Models and conditioning](models/linear-gaussian.md) or
[Particle Gibbs and Turing](inference.md).

## Shared model interface

These definitions belong to SSMProblems and are re-exported by GeneralisedFilters.

```@autodocs
Modules = [GeneralisedFilters.SSMProblems]
Order = [:type, :function]
```

## Models and inference algorithms

```@autodocs
Modules = [GeneralisedFilters]
Order = [:type, :function]
```

## Model validation utilities

These utilities help test custom process implementations against the model interface.

```@autodocs
Modules = [GeneralisedFilters.GFTest]
Order = [:type, :function]
```
