export ForwardAlgorithm, FW

struct ForwardAlgorithm <: AbstractFilter end
const FW = ForwardAlgorithm

function initialise(
    rng::AbstractRNG, model::DiscreteStateSpaceModel, ::ForwardAlgorithm; kwargs...
)
    return calc_α0(model.dyn; kwargs...)
end

function predict(
    rng::AbstractRNG,
    model::DiscreteStateSpaceModel{T},
    filter::ForwardAlgorithm,
    step::Integer,
    states::AbstractVector;
    kwargs...,
) where {T}
    P = calc_P(model.dyn, step; kwargs...)
    return (states' * P)'
end

function update(
    model::DiscreteStateSpaceModel{T},
    filter::ForwardAlgorithm,
    step::Integer,
    states::AbstractVector,
    observation;
    kwargs...,
) where {T}
    # Compute emission probability vector
    # TODO: should we define density as part of the interface or run the whole algorithm in
    # log space?
    b = map(
        x -> exp(SSMProblems.logdensity(model.obs, step, x, observation; kwargs...)),
        eachindex(states),
    )
    filtered_states = b .* states
    likelihood = sum(filtered_states)
    return (filtered_states / likelihood), log(likelihood)
end
