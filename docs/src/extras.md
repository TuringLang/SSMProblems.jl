# Control Variables and Extras

All functions that form part of the `SSMProblems` model interface demand that a final
positional argument called `extra` is included.

This argument has multiple uses, but is generally used to pass in additional
information to the model at inference time. Although this might seem unnecessary
and clunky for simple models, this addition leads to a great amount of
flexibility which allows complex and exotic models to be implemented with little
effort or performance penalty.

If your model does not require any extras, you can simply using `Nothing` as the
type for this argument.

When forward-simulating, filtering or smoothing from a model, a vector a
`extra`s is passed to the sampler, with each element corresponding to the
`extra` argument for each timestep. Some advanced algorithms may also augment
the `extra` vector with additional information.

## Use as Control Variables

In simple cases `extra` can be treated as a control (or input) vector. For
example, for data arriving at irregular time intervals, the `extra` argument
could be the time deltas between observations. Or, in control engineering, the
`extra` argument could be the [control inputs](https://www.mathworks.com/help/control/ref/ss.html) to the system.

Note, that it is also possible to store this data in the latent dynamic's struct
and extract it during a transition (e.g. `dyn.dts[timestep]`). However, this
approach  has the disadvantage that the control variables must be defined when
the model is instantiated. Further, this means that re-runs with new control
variables require a re-instantiation of the model.

Using `extra` for control variables allows for a separation between the abstract
definition of the state space model and the concrete simulation or inference
given specific data.

## Use with Streaming Data

The de-coupling of model definition and data that comes from using `extra` makes
it easy to use `SSMProblems` with streaming data. As control variables arrive,
these can be passed to the model distributions via the `extra` argument.

## Use in Rao-Blackwellisation

Briefly, a Rao-Blackwellised particle filter is an efficient variant of the
generic particle filter that can be applied to state space models that have an
analytically tractable sub-model. The filter behaves as two nested filters, a
regular particle filter for the outer model, and an analytic filter (e.g. Kalman
filter) for the inner sub-model.

Since the value of the `extra` argument can be defined at inference time, the
outer filter can pass information to the inner filter via the `extra` argument.
This leads to a clean and generic interface for Rao-Blackwellised filtering
which is not possible with other state space model packages.
