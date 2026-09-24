import Distributions: ContinuousMultivariateDistribution, _logpdf, _rand!
export SSMTrajectory

"""
    SSMTrajectory(model, [analytical_filter,] observations)

Turing boundary for a fixed-dimensional continuous trajectory. Its `logpdf` is the
joint trajectory/observation density, not a normalised trajectory prior. Include it exactly
once in the model; do not separately add the observations' likelihood. With an analytical
filter the trajectory stores outer states only and integrates out the inner states.

The trajectory is flattened for Turing storage. Its state dimensions must be constant over
time and parameters. Discrete and variable-dimensional trajectories are not supported by
this continuous-vector boundary. Standalone CSMC does not have this restriction.
"""
struct SSMTrajectory{MT<:StateSpaceModel,AFT,YT} <: ContinuousMultivariateDistribution
    model::MT
    af::AFT
    observations::YT
end
function SSMTrajectory(model::StateSpaceModel, observations)
    return SSMTrajectory(model, nothing, observations)
end

_state_dim(d::SSMTrajectory) = length(distribution(d.model.prior))
_state_dim(d::SSMTrajectory{<:HierarchicalSSM}) = length(distribution(d.model.prior.outer))
function _state_dim(d::SSMTrajectory{<:HierarchicalSSM,Nothing})
    p = d.model.prior
    x = mean(distribution(p.outer))
    return length(x) + length(distribution(inner_prior(p, x)))
end
Base.length(d::SSMTrajectory) = (length(d.observations) + 1) * _state_dim(d)
Base.eltype(::Type{<:SSMTrajectory}) = Float64

function _flatten_trajectory(traj, T::Integer, D::Integer)
    length(traj) == T + 1 || throw(DimensionMismatch("trajectory horizon mismatch"))
    all(t -> length(_trajectory_state(traj, t)) == D, 0:T) ||
        throw(DimensionMismatch("trajectory state dimension mismatch"))
    return reduce(vcat, [_trajectory_state(traj, t) for t in 0:T])
end
function _flatten_trajectory(
    traj::AbstractVector{<:HierarchicalState}, T::Integer, D::Integer
)
    length(traj) == T + 1 || throw(DimensionMismatch("trajectory horizon mismatch"))
    all(
        t ->
            length(_trajectory_state(traj, t).x) + length(_trajectory_state(traj, t).z) ==
            D,
        0:T,
    ) || throw(DimensionMismatch("trajectory state dimension mismatch"))
    return reduce(
        vcat,
        [vcat(_trajectory_state(traj, t).x, _trajectory_state(traj, t).z) for t in 0:T],
    )
end
_restore_state(template::Number, values) = only(values)
function _restore_state(template::StaticVector, values)
    return similar_type(typeof(template), eltype(values))(values)
end
_restore_state(template::AbstractVector, values) = values
_state_template(d::SSMTrajectory) = mean(distribution(d.model.prior))
function _state_template(d::SSMTrajectory{<:HierarchicalSSM})
    return mean(distribution(d.model.prior.outer))
end

function _trajectory_states(d::SSMTrajectory, flat)
    length(flat) == length(d) ||
        throw(DimensionMismatch("incorrect flattened trajectory length"))
    D = _state_dim(d)
    template = _state_template(d)
    return [
        _restore_state(template, flat[(t * D + 1):((t + 1) * D)]) for
        t in 0:length(d.observations)
    ]
end
function _trajectory_states(d::SSMTrajectory{<:HierarchicalSSM,Nothing}, flat)
    length(flat) == length(d) ||
        throw(DimensionMismatch("incorrect flattened trajectory length"))
    D = _state_dim(d)
    Dx = length(distribution(d.model.prior.outer))
    outer_template = _state_template(d)
    x0 = _restore_state(outer_template, flat[1:Dx])
    inner_template = mean(distribution(inner_prior(d.model, x0)))
    return [
        HierarchicalState(
            _restore_state(outer_template, flat[(t * D + 1):(t * D + Dx)]),
            _restore_state(inner_template, flat[(t * D + Dx + 1):((t + 1) * D)]),
        ) for t in 0:length(d.observations)
    ]
end
function _logpdf(d::SSMTrajectory, flat::AbstractVector{<:Real})
    states = _trajectory_states(d, flat)
    return if isnothing(d.af)
        trajectory_logdensity(d.model, states, d.observations)
    else
        trajectory_logdensity(d.model, d.af, states, d.observations)
    end
end
_flat_state(s) = s
_flat_state(s::HierarchicalState) = vcat(s.x, s.z)

function _rand!(rng::AbstractRNG, d::SSMTrajectory, flat::AbstractVector{<:Real})
    length(flat) == length(d) ||
        throw(DimensionMismatch("incorrect flattened trajectory length"))
    # Draw only the states represented by this factor. Marginalised inner states and
    # observed emissions need not support simulation (or nonsingular covariances).
    rb = d.model isa HierarchicalSSM && !isnothing(d.af)
    prior = rb ? d.model.prior.outer : d.model.prior
    dynamics = rb ? d.model.dyn.outer : d.model.dyn
    D = _state_dim(d)
    state = simulate(rng, prior)
    values = _flat_state(state)
    length(values) == D || throw(DimensionMismatch("trajectory state dimension mismatch"))
    flat[1:D] .= values
    for t in eachindex(d.observations)
        state = simulate(rng, dynamics, t, state)
        values = _flat_state(state)
        length(values) == D ||
            throw(DimensionMismatch("trajectory state dimension mismatch"))
        flat[(t * D + 1):((t + 1) * D)] .= values
    end
    return flat
end

function SSMTrajectory(model::AbstractStateSpaceModel, ys)
    return SSMTrajectory(StateSpaceModel(model), ys)
end
function SSMTrajectory(model::AbstractStateSpaceModel, af, ys)
    return SSMTrajectory(StateSpaceModel(model), af, ys)
end
