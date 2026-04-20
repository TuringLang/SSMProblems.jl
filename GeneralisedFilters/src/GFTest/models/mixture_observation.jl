"""
    MixtureObservation

Simple observation process for discrete state HMMs that emits Gaussian observations
with state-dependent means and unit variance.

Used in discrete filter/smoother tests where each discrete state k has mean μs[k].
"""
struct MixtureObservation{T<:Real,MT<:AbstractVector{T}} <: ObservationProcess
    μs::MT
end

function SSMProblems.logdensity(
    obs::MixtureObservation{T}, ::Integer, state::Integer, observation; kwargs...
) where {T}
    return logpdf(Normal(obs.μs[state], one(T)), observation)
end

function SSMProblems.distribution(
    obs::MixtureObservation{T}, ::Integer, state::Integer; kwargs...
) where {T}
    return Normal(obs.μs[state], one(T))
end
