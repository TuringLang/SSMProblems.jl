# SSMProblems

### Installation
In the `julia` REPL:
```julia
]add SSMProblems
```

### Documentation

`SSMProblems` defines a generic interface for State Space Problems (SSM). The main objective is to provide a consistent
interface to work with SSMs and their logdensities.

Consider a markovian model from[^Murray]:
![state space model](images/state_space_model.png)

[^Murray]:
    > Murray, Lawrence & Lee, Anthony & Jacob, Pierre. (2013). Rethinking resampling in the particle filter on graphics processing units. 

The model is fully specified by the following densities:
- __Initialisation__: ``f_0(x)``
- __Transition__: ``f(x)``
- __Emission__: ``g(x)``

The dynamics of the model are reduced to:
```math
\begin{aligned}
x_t | x_{t-1} &\sim f(x_t | x_{t-1}) \\
y_t | x_t &\sim g(y_t | x_{t})
\end{aligned}
```
assuming ``x_0 \sim f_0(x)``. 

The joint law follows:

```math
p(x_{0:T}, y_{0:T}) = f_0(x_0) \prod_t g(y_t | x_t) f(x_t | x_{t-1})
```

Users can define their SSM with `SSMProblems` in the following way:
```julia
struct Model <: AbstractStateSpaceModel end

# Define the structure of the latent space
particleof(::Model) = Float64
dimension(::Model) = 2

function transition!!(
    rng::Random.AbstractRNG, 
    step, 
    model::Model, 
    particle::AbstractParticl{<:AbstractStateSpaceModel}
) 
    if step == 1
        ... # Sample from the initial density
    end
    ... # Sample from the transition density
end

function emission_logdensity(step, model::Model, particle::AbstractParticle) 
    ... # Return log density of the model at *time* `step`
end

isdone(step, model::Model, particle::AbstractParticle) = ... # Stops the state machine

# Optionally, if the transition density is known, the model can also specify it
function transition_logdensity(step, prev_particle::AbstractParticle, next_particle::AbstractParticle)
    ... # Scores the forward transition at `x`
end
```

Package users can then consume the model `logdensity` through calls to `emission_logdensity`.  

For example, a bootstrap filter targeting the filtering distribution ``p(x_t | y_{0:t})`` using `N` particles would roughly follow:
```julia
struct Particle{T<:AbstractStateSpaceModel} <: AbstractParticle{T} end 

while !all(map(particle -> isdone(t, model, particles), particles)):
    ancestors = resample(rng, logweigths)
    particles = particles[ancestors]
    for i in 1:N
        particles[i] = transition!!(rng, t, model, particles[i])
        logweights[i] += emission_logdensity(t, model, particles[i])
    end
end
```

### Interface
```@autodocs
Modules = [SSMProblems]
Order   = [:type, :function]
```
