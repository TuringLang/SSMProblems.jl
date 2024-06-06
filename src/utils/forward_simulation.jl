"""Forward simulation of state space models."""

function StatsBase.sample(
    rng::AbstractRNG, model::AbstractStateSpaceModel, extras::AbstractVector
)
    T = length(extras)

    x0 = initialise(rng, model; extra=extras[1])
    y0 = observation(rng, model; state=x0, extra=extras[1])

    xs = Vector{eltype(x0)}(undef, T)
    ys = Vector{eltype(y0)}(undef, T)

    xs[1] = x0
    ys[1] = y0

    for t in 2:T
        xs[t] = transition(rng, model; state=xs[t - 1], step=t, extra=extras[t])
        ys[t] = observation(rng, model; state=xs[t], step=t, extra=extras[t])
    end

    return xs, ys
end

function StatsBase.sample(rng::AbstractRNG, model::AbstractStateSpaceModel, T::Integer)
    extras = [nothing for _ in 1:T]
    return sample(rng, model, extras)
end
