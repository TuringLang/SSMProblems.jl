export ForwardAlgorithm, filter

struct ForwardAlgorithm <: FilteringAlgorithm end

function initialise(model::DiscreteStateSpaceModel{T}, ::ForwardAlgorithm, extra) where {T}
    return calc_α0(model.dyn, extra)
end

function predict(
    model::DiscreteStateSpaceModel{T},
    ::ForwardAlgorithm,
    step::Integer,
    state::Vector,
    extra,
) where {T}
    P = calc_P(model.dyn, step, extra)
    return (state' * P)'
end

function update(
    model::DiscreteStateSpaceModel{T},
    ::ForwardAlgorithm,
    step::Integer,
    state::Vector,
    obs,
    extra,
) where {T}
    # Compute emission probability vector
    # TODO: should we define density as part of the interface or run the whole algorithm in
    # log space?
    b = [
        exp(SSMProblems.logdensity(model.obs, step, i, obs, extra)) for i in 1:length(state)
    ]
    filt_state = b .* state
    likelihood = sum(filt_state)
    return filt_state / likelihood, log(likelihood)
end

function step(
    model::DiscreteStateSpaceModel{T},
    filter::ForwardAlgorithm,
    step::Integer,
    state::Vector,
    obs,
    extra,
) where {T}
    state = predict(model, filter, step, state, extra)
    state, ll = update(model, filter, step, state, obs, extra)
    return state, ll
end

function filter(
    model::DiscreteStateSpaceModel{T},
    filter::ForwardAlgorithm,
    data::Vector,
    extra0,
    extras,
) where {T}
    state = initialise(model, filter, extra0)
    states = Vector{rb_eltype(model)}(undef, length(data))
    ll = 0.0
    for (i, obs) in enumerate(data[1:end])
        state, step_ll = step(model, filter, i, state, obs, extras[i])
        states[i] = state
        ll += step_ll
    end
    return states, ll
end
