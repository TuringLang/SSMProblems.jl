using SSMProblems

abstract type ConditionallyLinearGaussianSSM <: AbstractStateSpaceModel end

struct RaoBlackwellisedSSM
    conditioning_model::AbstractStateSpaceModel
    conditional_model::ConditionallyLinearGaussianSSM
end

###########################
#### CONDITIONAL MODEL ####
###########################

struct FullyLinearGaussianSubsetSSM <: ConditionallyLinearGaussianSSM
    """
        Consider a state space model with linear dynamics and Gaussian noise.
        The model is defined by the following equations:
        x[0] = z + ϵ,                 ϵ    ∼ N(0, P)
        x[k] = Φx[k-1] + b + w[k],    w[k] ∼ N(0, Q)
        y[k] = Hx[k] + v[k],          v[k] ∼ N(0, R)

        This SSM represents a subset of the state space model where some of the
        dimensions are conditionally independent given the rest.

        There are assumed to be `D1` conditioning dimensions, and that these
        are the first `D1` dimensions of the state vector.

        It is therefore required that the:
        - first `D1` columns of `H` are zero and that
        - upper-right `D1`x`D2` block of `Φ` is zero
        - `P`, `Q` are block diagonal
    """
    D1::Int  # Number of conditioning dimensions
    D2::Int  # Number of conditional dimensions
    z::Vector{Float64}
    P::Matrix{Float64}
    Φ::Matrix{Float64}
    b::Vector{Float64}
    Q::Matrix{Float64}
    H::Matrix{Float64}
    R::Matrix{Float64}
end

function transition!!(rng::AbstractRNG, model::FullyLinearGaussianSubsetSSM)
    D1 = model.D1
    return Gaussian(model.z[(D1 + 1):end], model.P[(D1 + 1):end, (D1 + 1):end])
end

function transition!!(
    rng::AbstractRNG,
    model::FullyLinearGaussianSubsetSSM,
    state::Gaussian,
    control::Vector{Float64},
)
    # Subset variables
    D1 = model.D1
    Φ_sub = model.Φ[(D1 + 1):end, (D1 + 1):end]
    b_sub = model.b[(D1 + 1):end]
    Q_sub = model.Q[(D1 + 1):end, (D1 + 1):end]

    # Conditioning (use previous conditioning state as a control variable)
    b_cond = model.Φ[(D1 + 1):end, 1:D1] * control

    # Transition
    μ = Φ_sub * state.μ + b_sub + b_cond
    Σ = Φ_sub * state.Σ * Φ_sub + Q_sub
    return Gaussian(μ, Σ)
end

function observation!!(
    rng::AbstractRNG, model::FullyLinearGaussianSubsetSSM, state::Gaussian
)
    return Gaussian(model.H * state.μ, model.R)
end

############################
#### CONDITIONING MODEL ####
############################

struct NonAnalyticLinearGaussianSSM <: AbstractStateSpaceModel
    """
        Consider a state space model with linear dynamics and Gaussian noise.
        The model is defined by the following equations:
        x[0] = z + ϵ,                 ϵ    ∼ N(0, P)
        x[k] = Φx[k-1] + b + w[k],    w[k] ∼ N(0, Q)
        y[k] = Hx[k] + v[k],          v[k] ∼ N(0, R)

        This SSM represents a subset of the state space model from which another
        SSM can be conditioned upon.

        There are assumed to be `D1` conditioning dimensions, and that these
        are the first `D1` dimensions of the state vector.

        It is therefore required that the:
        - first `D1` columns of `H` are zero and that
        - upper-right `D1`x`D2` block of `Φ` is zero
        - `P`, `Q` are block diagonal

        Although inference on this model can be performed analytically using the 
        Kalman filter, it is treated as a non-analytic model in this case for the
        purpose of testing.
    """
    D1::Int  # Number of conditioning dimensions
    D2::Int  # Number of conditional dimensions
    z::Vector{Float64}
    P::Matrix{Float64}
    Φ::Matrix{Float64}
    b::Vector{Float64}
    Q::Matrix{Float64}
    H::Matrix{Float64}
    R::Matrix{Float64}
end

function transition!!(rng::AbstractRNG, model::NonAnalyticLinearGaussianSSM)
    D1 = model.D1
    return rand(Gaussian(model.z[1:D1], model.P[1:D1, 1:D1]))
end

function transition!!(
    rng::AbstractRNG, model::NonAnalyticLinearGaussianSSM, state::Vector{Float64}
)
    D1 = model.D1
    Φ_sub = model.Φ[1:D1, 1:D1]
    b_sub = model.b[1:D1]
    Q_sub = model.Q[1:D1, 1:D1]

    return rand(Gaussian(Φ_sub * state + b_sub, Q_sub))
end

# NOTE: we do not need to define the emission_logdensity since the observations only
# depend on the conditioning state through the conditional variables
