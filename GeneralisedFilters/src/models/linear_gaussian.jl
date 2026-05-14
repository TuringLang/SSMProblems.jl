export GaussianPrior
export LinearGaussianLatentDynamics
export LinearGaussianObservationProcess
export LinearGaussianStateSpaceModel
export create_homogeneous_linear_gaussian_model
export hoist_static, step_params

export calc_μ0, calc_Σ0
export calc_A, calc_b, calc_Q
export calc_H, calc_c, calc_R
export calc_initial, calc_params

import SSMProblems: distribution, simulate_from_dist
import Distributions: MvNormal, params
import LinearAlgebra: cholesky
import Random: AbstractRNG, randn
import PDMats: PDMat, AbstractPDMat
using StaticArrays

# Custom sampling for MvNormal with static arrays to return SVector instead of Vector
function SSMProblems.simulate_from_dist(
    rng::AbstractRNG, d::MvNormal{T,<:AbstractPDMat{T},SVector{D,T}}
) where {T,D}
    μ, Σ = params(d)
    z = @SVector randn(rng, T, D)
    return μ + cholesky(Σ).L * z
end

## MODEL COMPONENTS ########################################################################

"""
    GaussianPrior(μ0, Σ0)

Multivariate Gaussian prior over the initial state. `μ0` and `Σ0` are
[`AbstractModelParameter`](@ref) wrappers; raw values are auto-wrapped via
[`as_parameter`](@ref).
"""
struct GaussianPrior{μP<:AbstractModelParameter,ΣP<:AbstractModelParameter} <: StatePrior
    μ0::μP
    Σ0::ΣP
end
GaussianPrior(μ0, Σ0) = GaussianPrior(as_parameter(μ0), as_parameter(Σ0))

"""
    LinearGaussianLatentDynamics(A, b, Q)

Linear-Gaussian latent dynamics `x_t = A x_{t-1} + b + w_t` with `w_t ~ N(0, Q)`. Each
field is an [`AbstractModelParameter`](@ref); raw matrices/vectors are auto-wrapped via
[`as_parameter`](@ref).
"""
struct LinearGaussianLatentDynamics{
    AP<:AbstractModelParameter,bP<:AbstractModelParameter,QP<:AbstractModelParameter
} <: LatentDynamics
    A::AP
    b::bP
    Q::QP
end
function LinearGaussianLatentDynamics(A, b, Q)
    return LinearGaussianLatentDynamics(as_parameter(A), as_parameter(b), as_parameter(Q))
end

"""
    LinearGaussianObservationProcess(H, c, R)

Linear-Gaussian observation process `y_t = H x_t + c + v_t` with `v_t ~ N(0, R)`. Each
field is an [`AbstractModelParameter`](@ref); raw matrices/vectors are auto-wrapped via
[`as_parameter`](@ref).
"""
struct LinearGaussianObservationProcess{
    HP<:AbstractModelParameter,cP<:AbstractModelParameter,RP<:AbstractModelParameter
} <: ObservationProcess
    H::HP
    c::cP
    R::RP
end
function LinearGaussianObservationProcess(H, c, R)
    return LinearGaussianObservationProcess(
        as_parameter(H), as_parameter(c), as_parameter(R)
    )
end

const LinearGaussianStateSpaceModel = StateSpaceModel{
    <:GaussianPrior,<:LinearGaussianLatentDynamics,<:LinearGaussianObservationProcess
}

function create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
    return SSMProblems.StateSpaceModel(
        GaussianPrior(μ0, Σ0),
        LinearGaussianLatentDynamics(A, b, Q),
        LinearGaussianObservationProcess(H, c, R),
    )
end

## HOIST / STEP-PARAM DISPATCH #############################################################

function hoist_static(p::GaussianPrior, θ, hoisted_controls)
    return (μ0=_hoist(p.μ0, θ, hoisted_controls), Σ0=_hoist(p.Σ0, θ, hoisted_controls))
end

function step_params(p::GaussianPrior, θ, hoisted_controls, hoist)
    return (
        μ0=_step_tagged(p.μ0, θ, 0, hoisted_controls, hoist.μ0),
        Σ0=_step_tagged(p.Σ0, θ, 0, hoisted_controls, hoist.Σ0),
    )
end

function hoist_static(c::LinearGaussianLatentDynamics, θ, hoisted_controls)
    return (
        A=_hoist(c.A, θ, hoisted_controls),
        b=_hoist(c.b, θ, hoisted_controls),
        Q=_hoist(c.Q, θ, hoisted_controls),
    )
end

function step_params(c::LinearGaussianLatentDynamics, θ, t, resolved, hoist)
    return (
        A=_step_tagged(c.A, θ, t, resolved, hoist.A),
        b=_step_tagged(c.b, θ, t, resolved, hoist.b),
        Q=_step_tagged(c.Q, θ, t, resolved, hoist.Q),
    )
end

function hoist_static(c::LinearGaussianObservationProcess, θ, hoisted_controls)
    return (
        H=_hoist(c.H, θ, hoisted_controls),
        c=_hoist(c.c, θ, hoisted_controls),
        R=_hoist(c.R, θ, hoisted_controls),
    )
end

function step_params(c::LinearGaussianObservationProcess, θ, t, resolved, hoist)
    return (
        H=_step_tagged(c.H, θ, t, resolved, hoist.H),
        c=_step_tagged(c.c, θ, t, resolved, hoist.c),
        R=_step_tagged(c.R, θ, t, resolved, hoist.R),
    )
end

## CALC-* SHIMS ############################################################################
#
# Transitional helpers used by non-gradient algorithm paths (predict, update, smoothers,
# RBPF). They evaluate a single parameter at time `t` using `kwargs` as the resolved
# controls. Only valid for parameters that do not depend on θ (Fixed and TimeVarying);
# FixedParametric / TimeVaryingParametric parameters must be evaluated through
# `ssm_loglikelihood`. These shims will be removed when the algorithms migrate to the
# `step_params` interface directly.

function _eval_param(p::AbstractModelParameter, t, kwargs)
    return _step_eval(p, nothing, t, NamedTuple(kwargs), _hoist(p, nothing, (;)))
end

calc_μ0(prior::GaussianPrior; kwargs...) = _eval_param(prior.μ0, 0, kwargs)
calc_Σ0(prior::GaussianPrior; kwargs...) = _eval_param(prior.Σ0, 0, kwargs)
function calc_initial(prior::GaussianPrior; kwargs...)
    return calc_μ0(prior; kwargs...), calc_Σ0(prior; kwargs...)
end

function calc_A(dyn::LinearGaussianLatentDynamics, t::Integer; kwargs...)
    return _eval_param(dyn.A, t, kwargs)
end
function calc_b(dyn::LinearGaussianLatentDynamics, t::Integer; kwargs...)
    return _eval_param(dyn.b, t, kwargs)
end
function calc_Q(dyn::LinearGaussianLatentDynamics, t::Integer; kwargs...)
    return _eval_param(dyn.Q, t, kwargs)
end
function calc_params(dyn::LinearGaussianLatentDynamics, t::Integer; kwargs...)
    return calc_A(dyn, t; kwargs...), calc_b(dyn, t; kwargs...), calc_Q(dyn, t; kwargs...)
end

function calc_H(obs::LinearGaussianObservationProcess, t::Integer; kwargs...)
    return _eval_param(obs.H, t, kwargs)
end
function calc_c(obs::LinearGaussianObservationProcess, t::Integer; kwargs...)
    return _eval_param(obs.c, t, kwargs)
end
function calc_R(obs::LinearGaussianObservationProcess, t::Integer; kwargs...)
    return _eval_param(obs.R, t, kwargs)
end
function calc_params(obs::LinearGaussianObservationProcess, t::Integer; kwargs...)
    return calc_H(obs, t; kwargs...), calc_c(obs, t; kwargs...), calc_R(obs, t; kwargs...)
end

## SSMPROBLEMS DISTRIBUTIONS ###############################################################

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
