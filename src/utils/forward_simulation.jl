"""Forward simulation of state space models."""

import AbstractMCMC: sample
export sample

function sample(rng::AbstractRNG, model::AbstractStateSpaceModel, extras::AbstractVector)
    T = length(extras)

    x0 = initialise(rng, model, extras[1])
    y0 = observation(rng, model, x0, 1, extras[1])

    xs = Vector{eltype(x0)}(undef, T)
    ys = Vector{eltype(y0)}(undef, T)

    xs[1] = x0
    ys[1] = y0

    for t in 2:T
        xs[t] = transition(rng, model, xs[t - 1], t, extras[t])
        ys[t] = observation(rng, model, xs[t], t, extras[t])
    end

    return xs, ys
end

function sample(rng::AbstractRNG, model::AbstractStateSpaceModel, T::Integer)
    extras = [nothing for _ in 1:T]
    return sample(rng, model, extras)
end
