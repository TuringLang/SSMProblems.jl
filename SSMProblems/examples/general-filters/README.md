# Filtering Interface Changes

This demo is a self contained proposal of the changes proposed for the GeneralizedFilters interface. While not fully featured, this example contains the relevant behaviors/consequences of the proposed SSMProblems changes.

## Filtering Structure

To ensure type stability, the filtering process can be broken into 3 parts:

1. sampling from the prior
2. instantiating intertemporal objects
3. iterating through observations

```julia
# [1] prior sampling
init_state = initialize(model, algo)

# [2] instantiation
state, log_evidence = filter_step(model, algo, 1, init_state, observations[1])

# [3] iteration
for t in 2:length(observations)
    state, log_marginal = filter_step(model, algo, t, state, observations[t])
    log_evidence += log_marginal
end
```

#### Initialization Versus Instantiation

Since we introduce a new element to SSMProblems, the `InitialStatePrior`, the type of the generated state via the prior is not always guaranteed to match with the dynamics.

Consider performing AD via ForwardDiff on a parameter found in the dynamics. Since prior samples are agnostic of the dynamics, any sampled states are not guaranteed to be of eltype `Dual`.

Given the user writes type stable dynamics/observation processes, separating these draws will ensure stability.

## Prediction Changes

The `predict` function operates quite differntly in this setting. Instead of baking the log incremental weights into the total log weights, they are part of the return similar to `update`.

While this change is *slightly* more memory intensive than past iterations, it ensures that the log evidence is accurate given a proposal.

Furthermore, it also implies that the interface is now fully consistent between the Kalman and particle filters. Since before, dispatch of `filter_step` was different for both algorithms due to this `marginalization_term`.

