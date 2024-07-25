"""
    Default sampling and log-density methods when corresponding distributions are defined.
"""

@default_arg function initialise(
    rng::AbstractRNG=default_rng(), dynamics::LatentDynamics, extra
)
    return rand(rng, initialisation_distribution(dynamics, extra))
end
function initialisation_logdensity(dynamics::LatentDynamics, state, extra)
    return logpdf(initialisation_distribution(dynamics, extra), state)
end

@default_arg function transition(
    rng::AbstractRNG=default_rng(), dynamics::LatentDynamics, state, step, extra
)
    return rand(rng, transition_distribution(dynamics, state, step, extra))
end
function transition_logdensity(dynamics::LatentDynamics, state, next_state, step, extra)
    return logpdf(transition_distribution(dynamics, state, step, extra), next_state)
end

@default_arg function observation(
    rng::AbstractRNG=default_rng(), process::ObservationProcess, state, step, extra
)
    return rand(rng, observation_distribution(process, state, step, extra))
end
function observation_logdensity(
    process::ObservationProcess, state, observation, step, extra
)
    return logpdf(observation_distribution(process, state, step, extra), observation)
end
