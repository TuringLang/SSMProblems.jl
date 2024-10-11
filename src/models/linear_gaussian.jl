export LinearGaussianLatentDynamics
export LinearGaussianObservationProcess
export LinearGaussianStateSpaceModel
export create_homogeneous_linear_gaussian_model

import SSMProblems: distribution
import Distributions: MvNormal

abstract type LinearGaussianLatentDynamics{T} <: SSMProblems.LatentDynamics{Vector{T}} end

function calc_μ0 end
function calc_Σ0 end
function calc_initial(dyn::LinearGaussianLatentDynamics; kwargs...)
    return calc_μ0(dyn; kwargs...), calc_Σ0(dyn; kwargs...)
end

function calc_A end
function calc_b end
function calc_Q end
function calc_params(dyn::LinearGaussianLatentDynamics, step::Integer; kwargs...)
    return (
        calc_A(dyn, step; kwargs...),
        calc_b(dyn, step; kwargs...),
        calc_Q(dyn, step; kwargs...)
    )
end

abstract type LinearGaussianObservationProcess{T} <:
              SSMProblems.ObservationProcess{Vector{T}} end

function calc_H end
function calc_c end
function calc_R end
function calc_params(obs::LinearGaussianObservationProcess, step::Integer; kwargs...)
    return (
        AnalyticalFilters.calc_H(obs, step; kwargs...),
        calc_c(obs, step; kwargs...),
        calc_R(obs, step; kwargs...),
    )
end

const LinearGaussianStateSpaceModel{T} = SSMProblems.StateSpaceModel{
    T,D,O
} where {T,D<:LinearGaussianLatentDynamics{T},O<:LinearGaussianObservationProcess{T}}

# TODO: this is hacky and should ideally be removed
# Can't use `eltype` because that is used by SSMProblems for forward simulation and would be
# used by a particle filtering.
function rb_eltype(::LinearGaussianStateSpaceModel{T}) where {T}
    return GaussianContainer{Vector{T},Matrix{T}} where {T}
end

#######################
#### DISTRIBUTIONS ####
#######################

function SSMProblems.distribution(
    dyn::LinearGaussianLatentDynamics;
    kwargs...,
)
    μ0, Σ0 = calc_initial(dyn; kwargs...)
    return MvNormal(μ0, Σ0)
end

function SSMProblems.distribution(
    dyn::LinearGaussianLatentDynamics{T},
    step::Integer,
    state::AbstractVector{T};
    kwargs...,
) where {T}
    A, b, Q = calc_params(dyn, step; kwargs...)
    return MvNormal(A * state + b, Q)
end

function SSMProblems.distribution(
    obs::LinearGaussianObservationProcess{T},
    step::Integer,
    state::AbstractVector{T};
    kwargs...,
) where {T}
    H, c, R = calc_params(obs, step; kwargs...)
    return MvNormal(H * state + c, R)
end

###########################################
#### HOMOGENEOUS LINEAR GAUSSIAN MODEL ####
###########################################

struct HomogeneousLinearGaussianLatentDynamics{T} <: LinearGaussianLatentDynamics{T}
    μ0::Vector{T}
    Σ0::Matrix{T}
    A::Matrix{T}
    b::Vector{T}
    Q::Matrix{T}
end
calc_μ0(dyn::HomogeneousLinearGaussianLatentDynamics; kwargs...) = dyn.μ0
calc_Σ0(dyn::HomogeneousLinearGaussianLatentDynamics; kwargs...) = dyn.Σ0
calc_A(dyn::HomogeneousLinearGaussianLatentDynamics, ::Integer; kwargs...) = dyn.A
calc_b(dyn::HomogeneousLinearGaussianLatentDynamics, ::Integer; kwargs...) = dyn.b
calc_Q(dyn::HomogeneousLinearGaussianLatentDynamics, ::Integer; kwargs...) = dyn.Q

struct HomogeneousLinearGaussianObservationProcess{T} <: LinearGaussianObservationProcess{T}
    H::Matrix{T}
    c::Vector{T}
    R::Matrix{T}
end
calc_H(obs::HomogeneousLinearGaussianObservationProcess, ::Integer; kwargs...) = obs.H
calc_c(obs::HomogeneousLinearGaussianObservationProcess, ::Integer; kwargs...) = obs.c
calc_R(obs::HomogeneousLinearGaussianObservationProcess, ::Integer; kwargs...) = obs.R

function create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
    return SSMProblems.StateSpaceModel(
        HomogeneousLinearGaussianLatentDynamics(μ0, Σ0, A, b, Q),
        HomogeneousLinearGaussianObservationProcess(H, c, R),
    )
end
