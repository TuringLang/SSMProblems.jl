export GaussianPrior
export LinearGaussianLatentDynamics
export LinearGaussianObservationProcess
export LinearGaussianStateSpaceModel
export create_homogeneous_linear_gaussian_model

import SSMProblems: distribution
import Distributions: MvNormal
import LinearAlgebra: cholesky

abstract type GaussianPrior <: StatePrior end

function calc_μ0 end
function calc_Σ0 end
function calc_initial(prior::GaussianPrior; kwargs...)
    return calc_μ0(prior; kwargs...), calc_Σ0(prior; kwargs...)
end

abstract type LinearGaussianLatentDynamics <: LatentDynamics end

function calc_A end
function calc_b end
function calc_Q end
function calc_params(dyn::LinearGaussianLatentDynamics, step::Integer; kwargs...)
    return (
        calc_A(dyn, step; kwargs...),
        calc_b(dyn, step; kwargs...),
        calc_Q(dyn, step; kwargs...),
    )
end

abstract type LinearGaussianObservationProcess <: ObservationProcess end

function calc_H end
function calc_c end
function calc_R end
function calc_params(obs::LinearGaussianObservationProcess, step::Integer; kwargs...)
    return (
        GeneralisedFilters.calc_H(obs, step; kwargs...),
        calc_c(obs, step; kwargs...),
        calc_R(obs, step; kwargs...),
    )
end

const LinearGaussianStateSpaceModel = StateSpaceModel{
    <:GaussianPrior,<:LinearGaussianLatentDynamics,<:LinearGaussianObservationProcess
}

function rb_eltype(model::LinearGaussianStateSpaceModel)
    μ0, Σ0 = calc_initial(model.prior)
    return Gaussian{typeof(μ0),typeof(Σ0)}
end

#######################
#### DISTRIBUTIONS ####
#######################

function SSMProblems.distribution(prior::GaussianPrior; kwargs...)
    μ0, Σ0 = calc_initial(prior; kwargs...)
    return MvNormal(μ0, Σ0)
end

function SSMProblems.distribution(
    dyn::LinearGaussianLatentDynamics, step::Integer, state::AbstractVector; kwargs...
)
    A, b, Q = calc_params(dyn, step; kwargs...)
    return MvNormal(A * state + b, Q)
end

function SSMProblems.distribution(
    obs::LinearGaussianObservationProcess, step::Integer, state::AbstractVector; kwargs...
)
    H, c, R = calc_params(obs, step; kwargs...)
    return MvNormal(H * state + c, R)
end

###########################################
#### HOMOGENEOUS LINEAR GAUSSIAN MODEL ####
###########################################

struct HomogeneousGaussianPrior{XT<:AbstractVector,ΣT<:AbstractMatrix} <: GaussianPrior
    μ0::XT
    Σ0::ΣT
end
calc_μ0(prior::HomogeneousGaussianPrior; kwargs...) = prior.μ0
calc_Σ0(prior::HomogeneousGaussianPrior; kwargs...) = prior.Σ0

struct HomogeneousLinearGaussianLatentDynamics{
    AT<:AbstractMatrix,bT<:AbstractVector,QT<:AbstractMatrix
} <: LinearGaussianLatentDynamics
    A::AT
    b::bT
    Q::QT
end
calc_A(dyn::HomogeneousLinearGaussianLatentDynamics, ::Integer; kwargs...) = dyn.A
calc_b(dyn::HomogeneousLinearGaussianLatentDynamics, ::Integer; kwargs...) = dyn.b
calc_Q(dyn::HomogeneousLinearGaussianLatentDynamics, ::Integer; kwargs...) = dyn.Q

struct HomogeneousLinearGaussianObservationProcess{
    HT<:AbstractMatrix,cT<:AbstractVector,RT<:AbstractMatrix
} <: LinearGaussianObservationProcess
    H::HT
    c::cT
    R::RT
end
calc_H(obs::HomogeneousLinearGaussianObservationProcess, ::Integer; kwargs...) = obs.H
calc_c(obs::HomogeneousLinearGaussianObservationProcess, ::Integer; kwargs...) = obs.c
calc_R(obs::HomogeneousLinearGaussianObservationProcess, ::Integer; kwargs...) = obs.R

function create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
    return SSMProblems.StateSpaceModel(
        HomogeneousGaussianPrior(μ0, Σ0),
        HomogeneousLinearGaussianLatentDynamics(A, b, Q),
        HomogeneousLinearGaussianObservationProcess(H, c, R),
    )
end

#######################
#### BATCH METHODS ####
#######################

function batch_calc_μ0s end
function batch_calc_Σ0s end
function batch_calc_As end
function batch_calc_bs end
function batch_calc_Qs end
function batch_calc_Hs end
function batch_calc_cs end
function batch_calc_Rs end

# TODO: can we remove batch size argument?
function batch_calc_initial(prior::GaussianPrior, N::Integer; kwargs...)
    return batch_calc_μ0s(prior, N; kwargs...), batch_calc_Σ0s(prior, N; kwargs...)
end

function batch_calc_params(
    dyn::LinearGaussianLatentDynamics, step::Integer, N::Integer; kwargs...
)
    return (
        batch_calc_As(dyn, step, N; kwargs...),
        batch_calc_bs(dyn, step, N; kwargs...),
        batch_calc_Qs(dyn, step, N; kwargs...),
    )
end

function batch_calc_params(
    obs::LinearGaussianObservationProcess, step::Integer, N::Integer; kwargs...
)
    return (
        batch_calc_Hs(obs, step, N; kwargs...),
        batch_calc_cs(obs, step, N; kwargs...),
        batch_calc_Rs(obs, step, N; kwargs...),
    )
end

function SSMProblems.batch_simulate(
    ::AbstractRNG, prior::HomogeneousGaussianPrior, N::Integer; kwargs...
)
    μ0, Σ0 = GeneralisedFilters.calc_initial(prior; kwargs...)
    Ls = repeat(cholesky(Σ0).L, 1, N)
    noise = CUDA.randn(T, length(μ0), N)
    return cu(μ0) .+ NNlib.batched_vec(Ls, noise)
end

function SSMProblems.batch_simulate(
    ::AbstractRNG,
    dyn::HomogeneousLinearGaussianLatentDynamics,
    step::Integer,
    prev_state;
    kwargs...,
)
    N = size(prev_state, 2)
    A, b, Q = calc_params(dyn, step; kwargs...)
    Ls = repeat(cholesky(Q).L, 1, N)
    As = repeat(A, 1, N)
    noise = CUDA.randn(T, length(b), N)
    return (NNlib.batched_vec(As, prev_state) .+ cu(b)) + NNlib.batched_vec(Ls, noise)
end

function batch_calc_μ0s(prior::HomogeneousGaussianPrior, N::Integer; kwargs...)
    return repeat(cu(prior.μ0), 1, N)
end
function batch_calc_Σ0s(prior::HomogeneousGaussianPrior, N::Integer; kwargs...)
    return repeat(cu(prior.Σ0), 1, N)
end
function batch_calc_As(
    dyn::HomogeneousLinearGaussianLatentDynamics, ::Integer, N::Integer; kwargs...
)
    return repeat(cu(dyn.A), 1, N)
end
function batch_calc_bs(
    dyn::HomogeneousLinearGaussianLatentDynamics, ::Integer, N::Integer; kwargs...
)
    return repeat(cu(dyn.b), 1, N)
end
function batch_calc_Qs(
    dyn::HomogeneousLinearGaussianLatentDynamics, ::Integer, N::Integer; kwargs...
)
    return repeat(cu(dyn.Q), 1, N)
end

function batch_calc_Hs(
    obs::HomogeneousLinearGaussianObservationProcess, ::Integer, N::Integer; kwargs...
)
    return repeat(cu(obs.H), 1, N)
end
function batch_calc_cs(
    obs::HomogeneousLinearGaussianObservationProcess, ::Integer, N::Integer; kwargs...
)
    return repeat(cu(obs.c), 1, N)
end

function batch_calc_Rs(
    obs::HomogeneousLinearGaussianObservationProcess, ::Integer, N::Integer; kwargs...
)
    return repeat(cu(obs.R), 1, N)
end
