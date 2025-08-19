"""Forward simulation of state space models."""

import AbstractMCMC: sample
export sample

"""
    sample([rng::AbstractRNG], model::StateSpaceModel, T::Integer; kwargs...)

Simulate a trajectory of length `T` from the state space model.

Returns a tuple `(x0, xs, ys)` where `x0` is the initial state, `xs` is a vector of latent states,
and `ys` is a vector of observations.
"""
function sample(
    rng::AbstractRNG, model::StateSpaceModel{<:Real,LD,OP}, T::Integer; kwargs...
) where {LD,OP}
    T_dyn = eltype(LD)
    T_obs = eltype(OP)

    xs = Vector{T_dyn}(undef, T)
    ys = Vector{T_obs}(undef, T)

    x0 = simulate(rng, model.dyn; kwargs...)
    for t in 1:T
        xs[t] = simulate(rng, model.dyn, t, t == 1 ? x0 : xs[t - 1]; kwargs...)
        ys[t] = simulate(rng, model.obs, t, xs[t]; kwargs...)
    end

    return x0, xs, ys
end

"""
    sample(model::AbstractStateSpaceModel, T::Integer; kwargs...)

Simulate a trajectory using the default random number generator.
"""
function sample(model::AbstractStateSpaceModel, T::Integer; kwargs...)
    return sample(default_rng(), model, T; kwargs...)
end