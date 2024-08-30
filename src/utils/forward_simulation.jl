"""Forward simulation of state space models."""

import AbstractMCMC: sample
export sample

function sample(rng::AbstractRNG, model::StateSpaceModel, extras::AbstractVector)
    T = length(extras)

    T1, T2 = eltype(model)
    xs = Vector{T1}(undef, T)
    ys = Vector{T2}(undef, T)

    x0 = simulate(rng, model.dyn)
    for t in 1:T
        xs[t] = simulate(rng, model.dyn, t, t == 1 ? x0 : xs[t - 1], extras[t])
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
