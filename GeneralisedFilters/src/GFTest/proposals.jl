using Distributions

"""Optimal importance proposal for a linear Gaussian model."""
struct OptimalProposal{D,O} <: AbstractProposal
    dyn::D
    obs::O
end
function GeneralisedFilters.distribution(prop::OptimalProposal, t::Integer, x, y)
    d = GeneralisedFilters.resolve(prop.dyn, (; t))
    o = GeneralisedFilters.resolve(prop.obs, (; t))
    state = GaussianState(d.A * x + d.b, d.Q)
    state, _ = GeneralisedFilters.kalman_update(state, o, y)
    return state
end

"""Transition proposal with covariance inflation, supporting ordinary and RB states."""
struct OverdispersedProposal{D} <: AbstractProposal
    dyn::D
    k::Float64
end
function GeneralisedFilters.distribution(prop::OverdispersedProposal, t::Integer, state, y)
    x = state isa RBState ? state.x : state
    d = GeneralisedFilters.resolve(prop.dyn, (; t))
    return GaussianState(d.A * x + d.b, prop.k * d.Q)
end
