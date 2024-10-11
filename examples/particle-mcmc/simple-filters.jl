using SSMProblems
using GaussianDistributions
using PDMats, LinearAlgebra
using Distributions
using Random
using UnPack
using StatsFuns

import AbstractMCMC: sample, AbstractSampler

## UTILITIES ###############################################################################

# GaussianDistributions.correct will error when type casting otherwise
function Base.convert(::Type{PDMat{T,MT}}, mat::MT) where {MT<:AbstractMatrix,T<:Real}
    return PDMat(Symmetric(mat))
end

# necessary for type stability of logpdf for Gaussian
function Distributions.logpdf(P::Gaussian, x)
    dP = length(P.μ)
    logdetcov = GaussianDistributions._logdet(P.Σ, dP)
    return -(
        GaussianDistributions.sqmahal(P, x) + logdetcov + dP * convert(eltype(x), log2π)
    ) / 2
end

## LINEAR GAUSSIAN STATE SPACE MODEL #######################################################

struct LinearGaussianLatentDynamics{
    T<:Real,AT<:AbstractMatrix{T},QT<:AbstractPDMat{T},ΣT<:AbstractPDMat{T}
} <: LatentDynamics{Vector{T}}
    """
        Latent dynamics for a linear Gaussian state space model.
        The model is defined by the following equations:
        x[t] = Ax[t-1] + b + ε[t],      ε[t] ∼ N(0, Q)
    """
    A::AT
    b::Vector{T}
    Q::QT

    init::Gaussian{Vector{T},ΣT}
end

# Convert covariance matrices to PDMats to avoid recomputing Cholesky factorizations
function LinearGaussianLatentDynamics(
    A::AbstractMatrix, b::Vector, Q::AbstractMatrix, init::Gaussian
)
    return LinearGaussianLatentDynamics(A, b, PDMat(Q), init)
end

function LinearGaussianLatentDynamics(
    A::AbstractMatrix, b::Vector, Q::AbstractVector, init::Gaussian
)
    return LinearGaussianLatentDynamics(A, b, PDiagMat(Q), init)
end

function LinearGaussianLatentDynamics(
    A::AbstractMatrix{T}, Q::AbstractArray{T}, init::Gaussian
) where {T<:Real}
    return LinearGaussianLatentDynamics(A, zeros(T, size(A, 1)), Q, init)
end

struct LinearGaussianObservationProcess{
    T<:Real,HT<:AbstractMatrix{T},RT<:AbstractPDMat{T}
} <: ObservationProcess{Vector{T}}
    """
        Observation process for a linear Gaussian state space model.
        The model is defined by the following equation:
        y[t] = Hx[t] + η[t],        η[t] ∼ N(0, R)
    """
    H::HT
    R::RT
end

function LinearGaussianObservationProcess(H::AbstractMatrix, R::AbstractMatrix)
    return LinearGaussianObservationProcess(H, PDMat(R))
end

function LinearGaussianObservationProcess(H::AbstractMatrix, R::AbstractVector)
    return LinearGaussianObservationProcess(H, PDiagMat(R))
end

function SSMProblems.distribution(
    proc::LinearGaussianLatentDynamics{T}; kwargs...
) where {T<:Real}
    dx = size(proc.A, 1)
    return MvNormal(proc.init.μ, proc.init.Σ)
end

function SSMProblems.distribution(
    proc::LinearGaussianLatentDynamics{T}, step::Int, state::AbstractVector{T}; kwargs...
) where {T<:Real}
    return MvNormal(proc.A * state + proc.b, proc.Q)
end

function SSMProblems.distribution(
    proc::LinearGaussianObservationProcess{T},
    step::Int,
    state::AbstractVector{T};
    kwargs...,
) where {T<:Real}
    return MvNormal(proc.H * state, proc.R)
end

const LinearGaussianModel{T} = StateSpaceModel{
    T,D,O
} where {T,D<:LinearGaussianLatentDynamics{T},O<:LinearGaussianObservationProcess{T}}

function PSDMat(mat::AbstractMatrix)
    # this deals with rank definicient positive semi-definite matrices
    chol_mat = cholesky(mat, Val(true); check=false)
    Up = UpperTriangular(chol_mat.U[:, invperm(chol_mat.p)])
    return PDMat(mat, Cholesky(Up))
end

## FILTERING ###############################################################################

abstract type AbstractFilter <: AbstractSampler end

"""
    predict([rng,] model, alg, step, states, [extra])

propagate the filtered states forward in time.
"""
function predict end

"""
    update(model, alg, step, states, data, [extra])

update beliefs on the propagated states.
"""
function update end

"""
    initialise([rng,] model, alg, [extra])

propose an initial state distribution.
"""
function initialise end

function sample(
    rng::AbstractRNG,
    model::StateSpaceModel,
    filter::AbstractFilter,
    observations::AbstractVector;
    callback=nothing,
    kwargs...,
)
    filtered_states = initialise(rng, model, filter; kwargs...)
    log_evidence = zero(eltype(model))

    for t in eachindex(observations)
        proposed_states = predict(rng, model, filter, t, filtered_states; kwargs...)

        filtered_states, log_marginal = update(
            model, filter, t, proposed_states, observations[t]; kwargs...
        )

        log_evidence += log_marginal

        callback === nothing ||
            callback(model, filter, t, filtered_states, observations; kwargs...)
    end

    return filtered_states, log_evidence
end

## KALMAN FILTER ###########################################################################

struct KalmanFilter <: AbstractFilter end

KF() = KalmanFilter()

function initialise(
    rng::AbstractRNG, model::LinearGaussianModel, filter::KalmanFilter; kwargs...
)
    init_dist = SSMProblems.distribution(model.dyn; kwargs...)
    return Gaussian(init_dist.μ, Matrix(init_dist.Σ))
end

function predict(
    rng::AbstractRNG,
    model::LinearGaussianModel,
    filter::KalmanFilter,
    step::Integer,
    states::Gaussian;
    kwargs...,
)
    @unpack A, b, Q = model.dyn

    predicted_states = let μ = states.μ, Σ = states.Σ
        Gaussian(A * μ + b, A * Σ * A' + Q)
    end

    return predicted_states
end

function update(
    model::LinearGaussianModel,
    filter::KalmanFilter,
    step::Integer,
    proposed_states::Gaussian,
    observation;
    kwargs...,
)
    @unpack H, R = model.obs

    states, residual, S = GaussianDistributions.correct(
        proposed_states, Gaussian(observation, R), H
    )

    log_marginal = logpdf(Gaussian(zero(residual), Symmetric(S)), residual)

    return (states, log_marginal)
end

## BOOTSTRAP FILTER ########################################################################

struct BootstrapFilter{T<:Real,RS<:AbstractResampler} <: AbstractFilter
    N::Int
    threshold::T
    resampler::RS
end

function BF(N::Integer; threshold::Real=1.0, resampler::AbstractResampler=Systematic())
    return BootstrapFilter(N, threshold, resampler)
end

resample_threshold(filter::BootstrapFilter) = filter.threshold * filter.N

function initialise(
    rng::AbstractRNG,
    model::StateSpaceModel,
    filter::BootstrapFilter;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    initial_states = map(x -> SSMProblems.simulate(rng, model.dyn; kwargs...), 1:(filter.N))
    initial_weights = zeros(eltype(model), filter.N)

    return update_ref!(ParticleContainer(initial_states, initial_weights), ref_state)
end

function resample(rng::AbstractRNG, states::ParticleContainer, filter::BootstrapFilter)
    weights = StatsBase.weights(states)
    ess = inv(sum(abs2, weights))
    @debug "ESS: $ess"

    if resample_threshold(filter) ≥ ess
        idx = resample(rng, filter.resampler, weights)
        reset_weights!(states)
    else
        idx = 1:(filter.N)
    end

    return idx
end

function predict(
    rng::AbstractRNG,
    model::StateSpaceModel,
    filter::BootstrapFilter,
    step::Integer,
    states::ParticleContainer{T};
    ref_state::Union{Nothing,AbstractVector{T}}=nothing,
    kwargs...,
) where {T}
    states.ancestors = resample(rng, states, filter)
    states.proposed = map(
        x -> SSMProblems.simulate(rng, model.dyn, step, x; kwargs...),
        states.filtered[states.ancestors],
    )

    return update_ref!(states, ref_state, step)
end

function update(
    model::StateSpaceModel,
    filter::BootstrapFilter,
    step::Integer,
    states::ParticleContainer,
    observation;
    kwargs...,
)
    log_marginals = map(
        x -> SSMProblems.logdensity(model.obs, step, x, observation; kwargs...),
        states.proposed,
    )

    prev_log_marginal = logsumexp(states.log_weights)
    states.log_weights += log_marginals
    states.filtered = states.proposed

    return (states, logsumexp(states.log_weights) - prev_log_marginal)
end
