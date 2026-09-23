# Recording filtering results

`filter` returns the final filtering state and the total log likelihood. To keep intermediate
results, write a loop with `initialise` and `step`, and append the results you need to a
container. There is no callback interface to configure.

For particle filters, choose between two containers:

- `ParticleTree` keeps the ancestry of the current particles and removes branches with no
  surviving descendants. It is useful when you need trajectories without retaining every
  past particle population.
- `DenseParticleContainer` keeps every population, its log weights and its ancestor indices.
  It uses more memory but lets you inspect individual filtering steps.

Both retain the initial states at time zero and the ancestor links from time one back to
those states. Particle counts stay fixed throughout a stored history.

## A manual particle-filtering loop

This example records both forms of history for a scalar autoregressive model. In practice,
choose the container that suits your application.

```@example history
using GeneralisedFilters, Distributions, Random

model = StateSpaceModel(
    DistributionPrior(Normal()),
    DistributionDynamics((t, x) -> Normal(0.9x, 0.2)),
    DistributionObservation((t, x) -> Normal(x, 0.3)),
)
observations = [0.2, -0.1, 0.3]

function record_filter(rng, model, algorithm, observations)
    isempty(observations) && throw(ArgumentError("observations must be nonempty"))
    initial = initialise(rng, model.prior, algorithm)
    tree = ParticleTree(initial)

    state, ll = GeneralisedFilters.step(rng, model, algorithm, 1, initial, observations[1])
    push!(tree, state)
    history = DenseParticleContainer(initial, state)
    for t in 2:length(observations)
        state, increment = GeneralisedFilters.step(rng, model, algorithm, t, state, observations[t])
        ll += increment
        push!(tree, state)
        push!(history, state)
    end
    return state, ll, tree, history
end

state, ll, tree, history = record_filter(MersenneTwister(42), model, BF(100), observations)
_, reference_ll = GeneralisedFilters.filter(MersenneTwister(42), model, BF(100), observations)
@assert ll ≈ reference_ll
@assert collect(get_ancestry(tree)[1]) == collect(get_ancestry(history, 1))
get_ancestry(tree)[1]
```

`get_ancestry(tree)` returns one `ReferenceTrajectory` for each current particle.
`get_ancestry(history, i)` follows the final particle at position `i` back through dense
history. Each trajectory has indices `0:T`. These are particle genealogies, not an
additional backward-smoothing calculation. For a Rao–Blackwellised filter, stored inner
states are the filtering distributions that accompanied those particles.

## When the initial state has a different type

`ParticleTree(initial)` assumes that later particle states have the same type as the initial
states. This is convenient for the example above. If the first transition or observation
changes the state representation, wait until the first completed step:

```julia
initial = initialise(rng, model.prior, algorithm)
first_state, ll = GeneralisedFilters.step(rng, model, algorithm, 1, initial, observations[1])
tree = ParticleTree(initial, first_state)
history = DenseParticleContainer(initial, first_state)
```

These constructors already store the first step, so start appending at time two. They infer
the later state type from `first_state`. The dense container also infers the numeric weight
type there, since an initial particle population may not yet have numeric weights.

A mismatched append raises an error before changing the stored history. If a one-argument
tree constructor chose the wrong type, the error points you to the two-argument constructor.
After the first step, state and weight types must stay consistent with the chosen container.
Invalid particle counts or ancestor indices are also rejected before pruning or appending.

## Ownership of mutable states

Containers copy the collection buffers used to store states, weights and ancestor indices.
Replacing an entry in an input vector will therefore not replace a stored entry. They do
**not** recursively copy each state object. A mutable vector or matrix inside a state remains
shared, including when you retrieve it through `get_ancestry`.

Once stored, treat a state object as read-only. A custom transition or update used with
history storage must leave earlier states unchanged. If your code reuses mutable state
buffers, explicitly store a snapshot instead:

```julia
push!(tree, deepcopy(state))
```

Similarly, use a copied initial population when constructing a history that must remain
independent of mutable inputs. Copy a retrieved trajectory before editing its mutable
states if the stored history must be preserved. Immutable states built from numbers and
StaticArrays need no such deep copies.

For analytical filters, a manual loop can collect the returned filtering distributions in
an ordinary vector. The same ownership rule applies to any mutable arrays they contain.
