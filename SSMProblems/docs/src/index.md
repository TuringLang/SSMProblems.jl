# SSMProblems

## Installation

In the `julia` REPL:

```julia
] add SSMProblems
```

## Documentation

`SSMProblems` defines a minimal interface for _state space models_ (SSMs). It defines three process abstract types, distribution adapters, the
`distribution`/`simulate`/`logdensity` generics, and a model interface with a standard
container.
Inference packages such as
[GeneralisedFilters](https://github.com/TuringLang/GeneralisedFilters.jl) build their
algorithms, parameter types and conditioning mechanisms on top of it, so a model written
against this interface can be used by algorithms that support its components.

Consider a standard (Markovian) state-space model from[^Murray]:
![state space model](images/state_space_model.png)

[^Murray]:
    > Murray, Lawrence & Lee, Anthony & Jacob, Pierre. (2013). Rethinking resampling in the particle filter on graphics processing units.

The following three distributions fully specify the model:

- The __initialisation__ distribution, ``f_0``, for the initial latent state ``X_0``
- The __transition__ distribution, ``f``, for the latent state ``X_t`` given the previous ``X_{t-1}``
- The __observation__ distribution, ``g``, for an observation ``Y_t`` given the state ``X_t``

The dynamics of the model are given by,

```math
\begin{aligned}
x_0 &\sim f_0(x_0) \\
x_t | x_{t-1} &\sim f(x_t | x_{t-1}) \\
y_t | x_t &\sim g(y_t | x_{t})
\end{aligned}
```

and the joint law is,

```math
p(x_{0:T}, y_{1:T}) = f_0(x_0) \prod_{t=1}^{T} g(y_t | x_t) f(x_t | x_{t-1}).
```

We can consider a state space model as being made up of two components:

- A latent Markov chain describing the evolution of the latent state
- An observation process describing the relationship between the latent states and the observations

Through this lens, we see that the distributions ``f_0``, ``f`` fully describe the latent Markov chain, whereas ``g`` describes the observation process.

A user of `SSMProblems` may define these three distributions directly. Alternatively, they
can define a subset of methods for sampling and evaluating log-densities of the
distributions, depending on the requirements of the filtering/smoothing algorithms they
intend to use.

For a model defined by distributions, use the adapters:

```@example model_interface
using Distributions, Random, SSMProblems

model = StateSpaceModel(
    DistributionPrior(Normal(0.0, 1.0)),
    DistributionDynamics((t, state) -> Normal(state, 0.1)),
    DistributionObservation((t, state) -> Normal(state, 0.5)),
)
x0, xs, ys = simulate(Xoshiro(42), model, 10)
```

The functions receive the time index, so time-varying coefficients can be captured in a
closure. For example, `DistributionDynamics((t, state) -> Normal(a[t] * state, 0.1))`
uses a vector of coefficients `a` from its enclosing scope.

You can also define a process type when you want to reuse it or provide specialised
methods. This dynamics component describes the same transition as the closure above:

```@example model_interface
struct RandomWalk <: LatentDynamics end
SSMProblems.distribution(::RandomWalk, t::Integer, state) = Normal(state, 0.1)
reusable_model = StateSpaceModel(prior(model), RandomWalk(), obs(model))
simulate(Xoshiro(42), reusable_model, 10)
```

There are a few things to note here:

- The prior takes no time or state arguments. The dynamics and observation process take
  both.
- Parameters, controls and external inputs are stored in the component or captured by its
  function. The process methods do not forward keyword arguments. Inference packages can
  provide further component types, such as linear-Gaussian transitions whose matrix
  structure can be used by a Kalman filter.
- If your latent dynamics or observation process cannot be represented as a `Distribution`
  object, implement `simulate` and/or `logdensity` directly instead, as documented below.

These distribution definitions are used to derive the `simulate` and `logdensity` methods
for each component. Package users then interact with the state space model through those
functions.

For example, a bootstrap filter targeting the filtering distribution ``p(x_t | y_{1:t})``
using `N` particles would roughly follow:

```julia
dynamics, observations_process = dyn(model), obs(model)

for (t, observation) in enumerate(observations)
    idx = resample(rng, log_weights)
    particles = particles[idx]
    fill!(log_weights, 0)
    for i in 1:N
        particles[i] = simulate(rng, dynamics, t, particles[i])
        log_weights[i] += logdensity(observations_process, t, particles[i], observation)
    end
end
```

For more thorough examples, see the provided example scripts.

### Custom model containers

`StateSpaceModel` stores the three components directly. A custom model container can
subtype `AbstractStateSpaceModel` and implement `prior(model)`, `dyn(model)` and
`obs(model)`. These accessors let simulation and inference code use the components without
requiring particular field names. `StateSpaceModel(model)` builds the standard container
from these accessors and retains the component objects.

`AbstractStateSpaceModel` has no AbstractMCMC dependency. Packages implementing samplers
can provide their own wrappers for that integration.

### Interface
```@autodocs
Modules = [SSMProblems]
Order   = [:type, :function, :module]
```
