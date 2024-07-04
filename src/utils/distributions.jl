"""
    Default sampling and log-density methods when corresponding distributions are defined.
"""

function initialise(rng::AbstractRNG, dynamics::LatentDynamics; extra)
    return rand(rng, initialisation_distribution(dynamics; extra))
end
function initialisation_logdensity(dynamics::LatentDynamics; state, extra)
    return logpdf(initialisation_distribution(dynamics; extra), state)
end

function transition(rng::AbstractRNG, dynamics::LatentDynamics; state, step, extra)
    return rand(rng, transition_distribution(dynamics; state, step, extra))
end
function transition_logdensity(dynamics::LatentDynamics; next_state, state, step, extra)
    return logpdf(transition_distribution(dynamics; state, step, extra), next_state)
end

function observation(rng::AbstractRNG, process::ObservationProcess; state, step, extra)
    return rand(rng, observation_distribution(process; state, step, extra))
end
function observation_logdensity(
    process::ObservationProcess; observation, state, step, extra
)
    return logpdf(observation_distribution(process; state, step, extra), observation)
end
