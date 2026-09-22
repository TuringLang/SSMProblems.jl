export GaussianPrior, LinearGaussianDynamics, LinearGaussianObservation
export DiscretePrior, DiscreteDynamics
export create_homogeneous_linear_gaussian_model

"""
    GaussianPrior(μ0, Σ0)

Gaussian initial-state prior with mean `μ0` and covariance `Σ0`, retaining both array
representations. Structured covariance matrices are supported.
"""
struct GaussianPrior{TM<:AbstractVector,TS<:AbstractMatrix} <: StatePrior
    μ0::TM
    Σ0::TS
end

"""
    LinearGaussianDynamics(A, b, Q)

Linear-Gaussian transition `x_t = A x_{t-1} + b + w`, `w ~ N(0, Q)`, with array
parameters, retaining the representation of `Q`.
"""
struct LinearGaussianDynamics{TA<:AbstractMatrix,Tb<:AbstractVector,TQ<:AbstractMatrix} <:
       LatentDynamics
    A::TA
    b::Tb
    Q::TQ
end

"""
    LinearGaussianObservation(H, c, R)

Linear-Gaussian emission `y_t = H x_t + c + v`, `v ~ N(0, R)`, with array parameters,
retaining the representation of `R`.
"""
struct LinearGaussianObservation{
    TH<:AbstractMatrix,Tc<:AbstractVector,TR<:AbstractMatrix
} <: ObservationProcess
    H::TH
    c::Tc
    R::TR
end

"""
    DiscretePrior(α0)

Initial distribution over a finite state space, with `α0[i] = p(x_0 = i)`.
"""
struct DiscretePrior{T<:AbstractVector} <: StatePrior
    α0::T
end

"""
    DiscreteDynamics(P)

Transition over a finite state space, with `P[i, j] = p(x_t = j | x_{t-1} = i)`.
"""
struct DiscreteDynamics{T<:AbstractMatrix} <: LatentDynamics
    P::T
end

# Atoms are constant processes: the time index is part of the interface but ignored here.
distribution(p::GaussianPrior) = GaussianState(p.μ0, p.Σ0)
distribution(d::LinearGaussianDynamics, ::Integer, z) = GaussianState(d.A * z + d.b, d.Q)
distribution(o::LinearGaussianObservation, ::Integer, z) = GaussianState(o.H * z + o.c, o.R)
distribution(p::DiscretePrior) = Categorical(p.α0)
distribution(d::DiscreteDynamics, ::Integer, i::Integer) = Categorical(d.P[i, :])

"""
    create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)

Convenience constructor for a time-invariant linear-Gaussian state-space model from constant
parameter arrays.
"""
function create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
    return StateSpaceModel(
        GaussianPrior(μ0, Σ0),
        LinearGaussianDynamics(A, b, Q),
        LinearGaussianObservation(H, c, R),
    )
end
