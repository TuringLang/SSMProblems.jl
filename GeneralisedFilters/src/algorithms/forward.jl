export ForwardAlgorithm, FW

struct ForwardAlgorithm <: AbstractFilter end
const FW = ForwardAlgorithm

function initialise(rng::AbstractRNG, prior::DiscretePrior, ::ForwardAlgorithm; kwargs...)
    return calc_α0(prior; kwargs...)
end

function predict(
    rng::AbstractRNG,
    dyn::DiscreteLatentDynamics,
    filter::ForwardAlgorithm,
    step::Integer,
    states::AbstractVector,
    observation;
    kwargs...,
)
    P = calc_P(dyn, step; kwargs...)
    return (states' * P)'
end

function update(
    obs::ObservationProcess,
    filter::ForwardAlgorithm,
    step::Integer,
    states::AbstractVector,
    observation;
    kwargs...,
)
    # Compute emission probability vector
    # TODO: should we define density as part of the interface or run the whole algorithm in
    # log space?
    b = map(
        x -> exp(SSMProblems.logdensity(obs, step, x, observation; kwargs...)),
        eachindex(states),
    )
    filtered_states = b .* states
    likelihood = sum(filtered_states)
    return (filtered_states / likelihood), log(likelihood)
end
