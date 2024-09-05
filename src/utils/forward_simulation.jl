"""Forward simulation of state space models."""

import AbstractMCMC: sample
export sample

function sample(rng::AbstractRNG, model::StateSpaceModel, extra0, extras::AbstractVector)
    T = length(extras)

    T_dyn, T_obs = eltype(model)
    xs = Vector{T_dyn}(undef, T)
    ys = Vector{T_obs}(undef, T)

    x0 = simulate(rng, model.dyn, extra0)
    for t in 1:T
        xs[t] = simulate(rng, model.dyn, t, t == 1 ? x0 : xs[t - 1], extras[t])
        ys[t] = simulate(rng, model.obs, t, xs[t], extras[t])
    end

    return x0, xs, ys
end
function sample(model::AbstractStateSpaceModel, extra0, extras::AbstractVector)
    return sample(default_rng(), model, extra0, extras)
end

function sample(rng::AbstractRNG, model::AbstractStateSpaceModel, T::Integer)
    return sample(rng, model, nothing, [nothing for _ in 1:T])
end
function sample(model::AbstractStateSpaceModel, T::Integer)
    return sample(default_rng(), model, T)
end
