"""
    Defines a dummy discrete model used for testing Rao-Blackwellised algorithms with
    discrete inner states.

    This file defines a hierarchical model where both outer and inner states are discrete,
    enabling exact comparison against a joint forward algorithm on the product state space.

    Structure:
    - Outer state: Discrete with K_outer states (sampled via particles)
    - Inner state: Discrete with K_inner states (analytically filtered)
    - Observations: Gaussian emissions depending on inner state

    The joint model is also provided as a standard discrete SSM on the product space
    with K_outer * K_inner states.
"""

export InnerDiscreteDynamics, DiscreteGaussianObservation, create_dummy_discrete_model

"""
    InnerDiscreteDynamics

Discrete dynamics for the inner state of a hierarchical model.
The transition matrix P_inner[i,j] = p(z_t = j | z_{t-1} = i).
Currently independent of the outer state.
"""
struct InnerDiscreteDynamics{PT<:AbstractMatrix} <: DiscreteLatentDynamics
    P::PT
end

GeneralisedFilters.calc_P(dyn::InnerDiscreteDynamics, ::Integer; kwargs...) = dyn.P

"""
    DiscreteGaussianObservation

Observation process that emits Gaussian observations based on discrete inner state.
Each discrete state k has associated mean μ[k] and variance σ²[k].
"""
struct DiscreteGaussianObservation{MT<:AbstractVector,VT<:AbstractVector} <:
       ObservationProcess
    μ::MT   # Mean for each state
    σ²::VT  # Variance for each state
end

function SSMProblems.distribution(
    obs::DiscreteGaussianObservation, ::Integer, state::Integer; kwargs...
)
    return Normal(obs.μ[state], sqrt(obs.σ²[state]))
end

function SSMProblems.logdensity(
    obs::DiscreteGaussianObservation, ::Integer, state::Integer, y; kwargs...
)
    return logpdf(Normal(obs.μ[state], sqrt(obs.σ²[state])), y)
end

"""
    JointDiscreteObservation

Observation process for the joint (product) state space.
Joint state index encodes (outer, inner) as: idx = (outer - 1) * K_inner + inner
"""
struct JointDiscreteObservation{MT<:AbstractVector,VT<:AbstractVector} <: ObservationProcess
    μ::MT
    σ²::VT
    K_inner::Int
end

function SSMProblems.distribution(
    obs::JointDiscreteObservation, ::Integer, joint_state::Integer; kwargs...
)
    inner_state = mod1(joint_state, obs.K_inner)
    return Normal(obs.μ[inner_state], sqrt(obs.σ²[inner_state]))
end

function SSMProblems.logdensity(
    obs::JointDiscreteObservation, ::Integer, joint_state::Integer, y; kwargs...
)
    inner_state = mod1(joint_state, obs.K_inner)
    return logpdf(Normal(obs.μ[inner_state], sqrt(obs.σ²[inner_state])), y)
end

"""
    create_dummy_discrete_model(rng, K_outer, K_inner; kwargs...)

Create a dummy hierarchical discrete model and its equivalent joint model.

Returns `(joint_model, hier_model)` where:
- `joint_model`: Standard discrete SSM on K_outer * K_inner product space
- `hier_model`: HierarchicalSSM with discrete outer and inner components

The joint model encodes (outer, inner) pairs as: idx = (outer - 1) * K_inner + inner
"""
function create_dummy_discrete_model(
    rng::AbstractRNG,
    K_outer::Integer,
    K_inner::Integer;
    obs_separation::Real=2.0,  # How separated the observation means are
    obs_noise::Real=0.5,       # Observation noise std dev
)
    # Generate random stochastic transition matrices
    P_outer = let M = rand(rng, K_outer, K_outer) .+ 0.1
        M ./ sum(M; dims=2)
    end

    P_inner = let M = rand(rng, K_inner, K_inner) .+ 0.1
        M ./ sum(M; dims=2)
    end

    # Initial distributions (uniform)
    α0_outer = fill(1.0 / K_outer, K_outer)
    α0_inner = fill(1.0 / K_inner, K_inner)

    # Observation parameters: well-separated means for each inner state
    μ_obs = collect(range(0.0; step=obs_separation, length=K_inner))
    σ²_obs = fill(obs_noise^2, K_inner)

    # Create hierarchical model
    outer_prior = HomogeneousDiscretePrior(α0_outer)
    outer_dyn = HomogeneousDiscreteLatentDynamics(P_outer)
    inner_prior = HomogeneousDiscretePrior(α0_inner)
    inner_dyn = InnerDiscreteDynamics(P_inner)
    obs = DiscreteGaussianObservation(μ_obs, σ²_obs)

    hier_model = HierarchicalSSM(outer_prior, outer_dyn, inner_prior, inner_dyn, obs)

    # Create joint model on product space
    K_joint = K_outer * K_inner

    # Joint initial distribution: α0_joint[(i-1)*K_inner + k] = α0_outer[i] * α0_inner[k]
    α0_joint = zeros(K_joint)
    for i in 1:K_outer, k in 1:K_inner
        α0_joint[(i - 1) * K_inner + k] = α0_outer[i] * α0_inner[k]
    end

    # Joint transition matrix
    # P_joint[(i,k) -> (j,l)] = P_outer[i,j] * P_inner[k,l]
    P_joint = zeros(K_joint, K_joint)
    for i in 1:K_outer, k in 1:K_inner
        idx_from = (i - 1) * K_inner + k
        for j in 1:K_outer, l in 1:K_inner
            idx_to = (j - 1) * K_inner + l
            P_joint[idx_from, idx_to] = P_outer[i, j] * P_inner[k, l]
        end
    end

    joint_prior = HomogeneousDiscretePrior(α0_joint)
    joint_dyn = HomogeneousDiscreteLatentDynamics(P_joint)
    joint_obs = JointDiscreteObservation(μ_obs, σ²_obs, K_inner)

    joint_model = StateSpaceModel(joint_prior, joint_dyn, joint_obs)

    return joint_model, hier_model
end
