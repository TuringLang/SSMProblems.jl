"""Forward simulation of state space models."""

using OffsetArrays: OffsetVector

import AbstractMCMC: sample

export sample

"""
    sample([rng::AbstractRNG], model::StateSpaceModel, T::Integer; kwargs...)

Simulate a trajectory of length `T` from the state space model.

Returns a tuple `(xs, ys)` where `xs` is an (offset) vector of latent states, and `ys` is a 
vector of observations. The latent states are indexed from `0` to `T`, where `xs[0]` is the
initial state.
"""
function sample(
    rng::AbstractRNG, model::StateSpaceModel{<:Real,LD,OP}, T::Integer; kwargs...
) where {LD,OP}
    T_dyn = eltype(LD)
    T_obs = eltype(OP)

    xs = OffsetVector(Vector{T_dyn}(undef, T + 1), -1)
    ys = Vector{T_obs}(undef, T)

    xs[0] = simulate(rng, model.dyn; kwargs...)
    for t in 1:T
        xs[t] = simulate(rng, model.dyn, t, xs[t - 1]; kwargs...)
        ys[t] = simulate(rng, model.obs, t, xs[t]; kwargs...)
    end

    return xs, ys
end

"""
    sample(model::AbstractStateSpaceModel, T::Integer; kwargs...)

Simulate a trajectory using the default random number generator.
"""
function sample(model::AbstractStateSpaceModel, T::Integer; kwargs...)
    return sample(default_rng(), model, T; kwargs...)
end
