"""Forward simulation of state space models."""

import AbstractMCMC: sample
export sample

function sample(rng::AbstractRNG, model::StateSpaceModel, extras::AbstractVector)
    T = length(extras)

    x0 = simulate(rng, model.dyn, extras[1])
    y0 = simulate(rng, model.obs, 1, x0, extras[1])

    xs = Vector{typeof(x0)}(undef, T)
    ys = Vector{typeof(y0)}(undef, T)

    xs[1] = x0
    ys[1] = y0

    for t in 2:T
        xs[t] = simulate(rng, model.dyn, t, xs[t - 1], extras[t])
        ys[t] = simulate(rng, model.obs, t, xs[t], extras[t])
    end

    return xs, ys
end
function sample(model::AbstractStateSpaceModel, extras::AbstractVector)
    return sample(default_rng(), model, extras)
end

function sample(rng::AbstractRNG, model::AbstractStateSpaceModel, T::Integer)
    extras = [nothing for _ in 1:T]
    return sample(rng, model, extras)
end
function sample(model::AbstractStateSpaceModel, T::Integer)
    return sample(default_rng(), model, T)
end
