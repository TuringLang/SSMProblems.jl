using GeneralisedFilters
using SSMProblems
using LinearAlgebra
using MatrixEquations

# adaoted from QuantEcon.jl
include("gensys.jl")

abstract type DSGE end

"""
A log linearized DSGE model which fits into the following form:
```
Γ0 * y[t] = Γ1 * y[t-1] + c + Ψ * z[t] + Π * η[t]
```
where z is a vector of exogenous shocks and η is a vector of noise from one-step-ahead forward
looking expectations.
"""
struct LogLinearizedModel{
    T <: Real,
    Γ0T <: AbstractMatrix{T},
    Γ1T <: AbstractMatrix{T},
    ΨT  <: AbstractMatrix{T},
    ΠT  <: AbstractMatrix{T},
    CT  <: AbstractVector{T}
} <: DSGE
    Γ0::Γ0T
    Γ1::Γ1T
    Ψ::ΨT
    Π::ΠT
    C::CT
end

function state_space(dsge::LogLinearizedModel; kwargs...)
    policy, drift, impact, eu = gensys(dsge.Γ0, dsge.Γ1, dsge.C, dsge.Ψ, dsge.Π; kwargs...)

    latent_states = Int64[]
    for (i, col) in enumerate(eachcol(policy))
        any(abs.(col) .> 1e-12) && push!(latent_states, i)
    end

    state_perm = permutation_matrix(policy, latent_states)
    obs_perm = permutation_matrix(policy, [1,2,3])

    # latent dynamics
    A = state_perm * policy * state_perm'
    B = state_perm * impact
    Q = B * B'
    b = state_perm * drift

    # observation process
    H = obs_perm * policy * state_perm'
    D = obs_perm * impact
    R = D * D'
    c = obs_perm * drift

    # initial state (assuming process is stationary)
    X0 = zero(b)
    Σ0 = lyapd(A, Q)

    return create_homogeneous_linear_gaussian_model(X0, Σ0, A, b, Q, H, c, R)
end

function submatrix(A, perm)
    P = permutation_matrix(A, perm)
    return P*A*P'
end

function submatrix(A, row_perm, col_perm)
    P1 = permutation_matrix(A, row_perm)
    P2 = permutation_matrix(A, col_perm)
    return P1*A*P2'
end

function permutation_matrix(A::Matrix{T}, perm::AbstractVector) where {T<:Number}
    P = zeros(T, (length(perm), size(A, 1)))
    for (i, p) in enumerate(perm)
        P[i, p] = one(T)
    end
    return P
end

#= model equations
\begin{align*}
    y_{t} &= y_{t+1} - 1 / \gamma (i_{t} - \pi_{t+1}) + \omega^{d}_{t} \\
    \pi_{t} &= \beta \pi_{t+1} + (1-\theta)*(1 - \theta \beta) / \theta * (\gamma + \varphi) y_{t} - \omega^{s}_{t} \\
    i_{t} &= \phi^{i} i_{t-1} + (1-\phi^{i})*(\phi^{\pi} \pi_{t} + \phi^{y} y_{t}) + \omega^{m}_{t}
\end{align*}
=#

#= shocks
\begin{align*}
    \omega^{d}_{t} &= \rho^{d} * \omega^{d}_{t-1} + \sigma^{d} \varepsilon^{d}_{t} \\
    \omega^{s}_{t} &= \rho^{s} * \omega^{s}_{t-1} + \sigma^{s} \varepsilon^{s}_{t} \\
    \omega^{m}_{t} &= \rho^{m} * \omega^{m}_{t-1} + \sigma^{m} \varepsilon^{m}_{t}
\end{align*}
=#

function new_keynesian_model(
    β::T, γ::T, φ::T, θ::T, ϕπ::T, ϕy::T, ϕi::T, ρd::T, ρs::T, ρm::T, σd::T, σs::T, σm::T
) where {T<:Real}
    Γ0 = zeros(T, (8, 8))
    Γ1 = zeros(T, (8, 8))
    Ψ  = zeros(T, (8, 3))
    Π  = zeros(T, (8, 2))
    C  = zeros(T, 8)

    # endogenous model equations
    Γ0[1,:] = [1 0 (1/γ) -1 -(1/γ) -1 0 0]
    Γ0[2,:] = [-(1-θ)*(1-θ*β)/θ*(γ+φ) 1 0 0 -β 0 1 0]
    Γ0[3,:] = [-(1-ϕi)*ϕy -(1-ϕi)*ϕπ 1 0 0 0 0 -1]

    Γ1[3,3] = ϕi

    # forward lookers
    Γ0[4,1] = one(T)
    Γ0[5,2] = one(T)

    Γ1[4,4] = one(T)
    Γ1[5,5] = one(T)

    Π[4:5,:] = I(2)

    # shock processes
    Γ0[6:end,6:end] = I(3)
    Γ1[6:end,6:end] = diagm([ρd, ρs, ρm])
    Ψ[6:end,:]  = diagm([σd, σs, σm])

    return LogLinearizedModel(Γ0, Γ1, Ψ, Π, C)
end

# create the DSGE model
dsge = new_keynesian_model(
    0.995, 1.0, 1.0, 0.75, 1.5, 0.1, 0.9, 0.8, 0.9, 0.2, 1.6013, 0.9488, 0.2290
)

# generate the state space form
model = state_space(dsge)

# query some FRED data and let it rip...
