"""
Methods for performing batch simulation and log-density evaluation for state space models.
"""

"""
    batch_simulate([rng::AbstractRNG], dyn::LatentDynamics, N::Integer; kwargs...)

    Simulate a batch of initial states for the latent dynamics.

    The method should return a batch of random initial states for the first time step of the
    latent dynamics. The batch size is determined by the `N` argument.

    See also [`LatentDynamics`](@ref).
"""
function batch_simulate(rng::AbstractRNG, dyn::LatentDynamics, N::Integer; kwargs...)
    throw(MethodError(batch_simulate, (rng, dyn, N, kwargs...)))
end

"""
    batch_simulate([rng::AbstractRNG], dyn::LatentDynamics, step::Integer, prev_states; kwargs...)

    Simulate a batch of transitions of the latent dynamics.

    The method should return a batch of random states for the current time step, `step`,
    given the previous states, `prev_states`.

    See also [`LatentDynamics`](@ref).
"""
function batch_simulate(
    rng::AbstractRNG, dyn::LatentDynamics, step::Integer, prev_states; kwargs...
)
    throw(MethodError(batch_simulate, (rng, dyn, step, prev_states, kwargs...)))
end

"""
    batch_simulate([rng::AbstractRNG], obs::ObservationProcess, step::Integer, states; kwargs...)
    
    Simulate a batch of observations given the current states.

    The method should return a batch of random observations given the current states
    `states` at time step `step`.

    See also [`ObservationProcess`](@ref).
"""
function batch_simulate(
    rng::AbstractRNG, obs::ObservationProcess, step::Integer, states; kwargs...
)
    throw(MethodError(batch_simulate, (rng, obs, step, states, kwargs...)))
end

"""
    batch_logdensity([rng::AbstractRNG], dyn::LatentDynamics, new_states; kwargs...)

    Compute the log-densities of a batch of initial states for the latent dynamics.

    The method should return the log-densities of a batch of initial states `new_states`
    for the initial time step of the latent dynamics.

    See also [`LatentDynamics`](@ref).
"""
function batch_logdensity(rng::AbstractRNG, dyn::LatentDynamics, new_states; kwargs...)
    throw(MethodError(batch_logdensity, (rng, dyn, new_states, kwargs...)))
end

"""
    batch_logdensity([rng::AbstractRNG], dyn::LatentDynamics, step::Integer, prev_states, new_states; kwargs...)

    Compute the log-densities of a batch of transitions of the latent dynamics.

    The method should return the log-densities of a batch of states `new_states` for the
    current time step, `step`, given the previous states, `prev_states`.

    See also [`LatentDynamics`](@ref).
"""
function batch_logdensity(
    rng::AbstractRNG, dyn::LatentDynamics, step::Integer, prev_states, new_states; kwargs...
)
    throw(
        MethodError(batch_logdensity, (rng, dyn, step, prev_states, new_states, kwargs...))
    )
end

"""
    batch_logdensity([rng::AbstractRNG], obs::ObservationProcess, step::Integer, state, observations; kwargs...)

    Compute the log-densities of a batch of observations given the current states.

    The method should return the log-densities of a batch of observations `observations`
    given the current states `states` at time step `step`.

    See also [`ObservationProcess`](@ref).
"""
function batch_logdensity(
    rng::AbstractRNG, obs::ObservationProcess, step::Integer, state, observations; kwargs...
)
    throw(MethodError(batch_logdensity, (rng, obs, step, state, observations, kwargs...)))
end
