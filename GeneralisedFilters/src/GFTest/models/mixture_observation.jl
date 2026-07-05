"""
    MixtureObservation

Observation process for discrete-state HMMs that emits Gaussian observations with
state-dependent means and unit variance. Used in discrete filter/smoother tests where each
discrete state `k` has mean `μs[k]`.
"""
struct MixtureObservation{T<:Real,MT<:AbstractVector{T}} <:
       GeneralisedFilters.ObservationProcess
    μs::MT
end

function GeneralisedFilters.logdensity(
    obs::MixtureObservation{T}, ::Integer, state::Integer, observation
) where {T}
    return logpdf(Normal(obs.μs[state], one(T)), observation)
end

function GeneralisedFilters.distribution(
    obs::MixtureObservation{T}, ::Integer, state::Integer
) where {T}
    return Normal(obs.μs[state], one(T))
end
