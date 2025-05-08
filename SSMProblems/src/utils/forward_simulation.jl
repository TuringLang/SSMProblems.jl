"""Forward simulation of state space models."""

using OffsetArrays: OffsetVector

import AbstractMCMC: sample
export sample

"""
    sample([rng::AbstractRNG], model::StateSpaceModel, T::Integer; kwargs...)

Simulate a trajectory of length `T` from the state space model.

Returns a tuple `(xs, ys)` where `xs` is a vector of latent states (including the initial
state) and `ys` is a vector of observations.
"""
function sample(rng::AbstractRNG, model::StateSpaceModel, T::Integer; kwargs...)
    xs = OffsetVector(fill(simulate(rng, model.prior), T + 1), -1)
    for t in 1:T
        xs[t] = simulate(rng, model.dyn, t, xs[t - 1]; kwargs...)
    end
    return xs, map(t -> simulate(rng, model.obs, t, xs[t]; kwargs...), 1:T)
end

"""
    sample(model::AbstractStateSpaceModel, T::Integer; kwargs...)

Simulate a trajectory using the default random number generator.
"""
function sample(model::AbstractStateSpaceModel, T::Integer; kwargs...)
    return sample(default_rng(), model, T; kwargs...)
end