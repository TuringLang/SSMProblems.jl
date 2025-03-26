using LinearAlgebra

abstract type DSGE end

"""
A log linearized DSGE model which fits into the following form:
```
Γ0 * y[t] = Γ1 * y[t-1] + c + Ψ * z[t] + Π * η[t]
```
where z is a vector of exogenous shocks and η is a vector of noise from one-step-ahead forward
looking expectations.
"""
struct LinearRationalExpectation{
    T<:Real,
    Γ0T<:AbstractMatrix{T},
    Γ1T<:AbstractMatrix{T},
    ΨT<:AbstractMatrix{T},
    ΠT<:AbstractMatrix{T},
    CT<:AbstractVector{T},
} <: DSGE
    Γ0::Γ0T
    Γ1::Γ1T
    Ψ::ΨT
    Π::ΠT
    C::CT
end

# takes advantage of upper triangular form of the QZ decomposition
function partition(A::UpperTriangular, n::Int)
    A11 = UpperTriangular(A[1:n, 1:n])
    A12 = A[1:n, (n + 1):end]
    A22 = UpperTriangular(A[(n + 1):end, (n + 1):end])
    return A11, A12, A22
end

function partition(A::AbstractMatrix, n::Int)
    A1 = A[1:n, :]
    A2 = A[(n + 1):end, :]
    return A1, A2
end

"""
this is an algorithm proposed by (Sims, 1995) which restructures linear rational expectation
models such that the noise generated from impact is no longer a function of expectations.
"""
function gensys(dsge::LinearRationalExpectation{T}; ϵ::T=1e-6) where {T<:Real}
    Π, Ψ, C = dsge.Π, dsge.Ψ, dsge.C
    F = schur(dsge.Γ0, dsge.Γ1)
    eigenvalues = F.β ./ F.α

    stable_flag = abs.(eigenvalues) .< 1 + ϵ
    nstable = count(stable_flag)
    ordschur!(F, stable_flag)

    # by definition Λ and Ω are upper triangular
    Λ, Ω = UpperTriangular(F.S), UpperTriangular(F.T)
    Z, Q = F.Z, F.Q'

    # partition the system
    Q1, Q2 = partition(Q, nstable)
    Z1, _ = transpose.(partition(Z', nstable))
    Λ11, Λ12, Λ22 = partition(Λ, nstable)
    Ω11, Ω12, Ω22 = partition(Ω, nstable)

    if size(Q2, 1) == size(Π, 2)
        Φ = Q1 * Π * inv(Q2 * Π)
    else
        @error "Blanchard Kahn conditions violated"
    end

    # premultiply according to (Sims, 1995)
    Λinv = inv(Λ11)
    n = length(stable_flag)
    H = Z * [Λinv (Λinv*(Λ12 - Φ * Λ22)); zeros(n - nstable, nstable) I]

    # solve via back substitution
    impact = H * [Q1 - Φ * Q2; zero(Π)'] * Ψ
    policy = Z1 * Λinv * [Ω11 (Ω12 − Φ * Ω22)] * Z'
    constant = H * [Q1 - Φ * Q2; inv(Ω22 - Λ22) * Q2] * C

    return impact, policy, constant
end

function SSMProblems.StateSpaceModel(model::LinearRationalExpectation; kwargs...)
    impact, policy, constant = gensys(model; kwargs...)

    latent_states = Int64[]
    for (i, col) in enumerate(eachcol(policy))
        any(abs.(col) .> 1e-12) && push!(latent_states, i)
    end

    state_perm = permutation_matrix(policy, latent_states)
    obs_perm = permutation_matrix(policy, [1, 2, 3])

    # latent dynamics
    A = state_perm * policy * state_perm'
    B = state_perm * impact
    Q = B * B'
    b = state_perm * constant

    # observation process
    H = obs_perm * policy * state_perm'
    D = obs_perm * impact
    R = D * D'
    c = obs_perm * constant

    # initial state (assuming process is stationary)
    X0 = zero(b)
    Σ0 = lyapd(A, Q)

    return create_homogeneous_linear_gaussian_model(X0, Σ0, A, b, Q, H, c, R)
end

function submatrix(A, perm)
    P = permutation_matrix(A, perm)
    return P * A * P'
end

function submatrix(A, row_perm, col_perm)
    P1 = permutation_matrix(A, row_perm)
    P2 = permutation_matrix(A, col_perm)
    return P1 * A * P2'
end

function permutation_matrix(A::Matrix{T}, perm::AbstractVector) where {T<:Number}
    P = zeros(T, (length(perm), size(A, 1)))
    for (i, p) in enumerate(perm)
        P[i, p] = one(T)
    end
    return P
end