# Control Variables and Extras

All functions that form part of the `SSMProblems` model interface should accept
keyword arguments.

These argument has multiple uses, but are generally used to pass in additional
information to the model at inference time. Although this might seem unnecessary
and clunky for simple models, this addition leads to a great amount of
flexibility that allows complex and exotic models to be implemented with little
effort or performance penalty.

If your model does not require any keyword arguments, you do not need to use any
in your function body (though `; kwargs...` should still be included in the signature).

When forward-simulating, filtering or smoothing from a model, these keyword
arguments are passed to the SSM definition. Some advanced algorithms such as the
Rao-Blackwellised particle filter may also introduce additional keyword
arguments at inference time.

## Use as Control Variables

In simple cases `kwargs` can be used to specify a control (or input) vector as
is common in [control engineering](https://www.mathworks.com/help/control/ref/ss.html).

In this case, the `simulate` function for the latent dynamics may look like
this:

```julia
function simulate(
    rng::AbstractRNG, 
    dyn::SimpleLatentDynamics, 
    step::Int, 
    state::Float64; 
    control::Float64,  # new keyword argument
    kwargs...
)
    return state + control + rand(rng, Normal(0.0, 0.1))
end
```

## Use as Time Deltas

Keywords are not limited to be used as simple control vectors, and can in fact
be used to pass in arbitrary additional information to the model at runtime. A
common use case is when considering data arriving at irregular time intervals.
In this case, they keyword arguments can be used to pass in the time delta
between observations.

In this case, the `simulate` function for the latent dynamics may look like
this: 

```julia
function simulate(
    rng::AbstractRNG, 
    dyn::SimpleLatentDynamics, 
    step::Int, 
    state::Float64; 
    dts::Vector{Float64},  # new keyword argument
    kwargs...
)
    dt = dts[step]
    return state + dt * rand(rng, Normal(0.1, 1.0))
end
```

Note, that it is also possible to store this data in the latent dynamic's struct
and extract it during a transition (e.g. `dyn.dts[timestep]`). However, this
approach  has the disadvantage that the control variables must be defined when
the model is instantiated. Further, this means that re-runs with new control
variables require a re-instantiation of the model.

Using keyword arguments for control variables allows for a separation between
the abstract definition of the state space model and the concrete simulation or
inference given specific data.

## Use with Streaming Data

The de-coupling of model definition and data that comes from using keyword
arguments makes it easy to use `SSMProblems` with streaming data. As control
variables arrive, these can be passed to the model distributions via keyword arguments.

## Use in Rao-Blackwellisation

Briefly, a Rao-Blackwellised particle filter is an efficient variant of the
generic particle filter that can be applied to state space models that have an
analytically tractable sub-model. The filter behaves as two nested filters, a
regular particle filter for the outer model, and an analytic filter (e.g. Kalman
filter) for the inner sub-model.

Since the value of the keyword arguments can be defined at inference time, the
outer filter can pass information to the inner filter via through these.
This leads to a clean and generic interface for Rao-Blackwellised filtering,
which is not possible with other state space model packages.
