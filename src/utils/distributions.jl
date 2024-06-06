"""
    Default sampling and log-density methods when corresponding distribution are defined.
"""

function check_initial_distribution(dynamics::LatentDynamics)
    if !hasmethod(initialisation_distribution, Tuple{typeof(dynamics),Integer})
        calling_func = StackTraces.stacktrace()[2].func
        error(
            "neither $calling_func nor initialisation_distribution is defined for $dynamics"
        )
    end
end
function initialise(rng::AbstractRNG, dynamics::LatentDynamics; extra)
    check_initial_distribution(dynamics)
    return rand(rng, initialisation_distribution(dynamics; extra))
end
function initialisation_logdensity(dynamics::LatentDynamics; state, extra)
    check_initial_distribution(dynamics)
    return logpdf(initialisation_distribution(dynamics; extra), state)
end

function check_transition_distribution(dynamics::LatentDynamics)
    if !hasmethod(
        transition_distribution,
        Tuple{typeof(dynamics),AbstractVector,AbstractVector,Integer},
    )
        calling_func = StackTraces.stacktrace()[2].func
        error("neither $calling_func nor transition_distribution is defined for $dynamics")
    end
end
function transition(rng::AbstractRNG, dynamics::LatentDynamics; state, step, extra)
    check_transition_distribution(dynamics)
    return rand(rng, transition_distribution(dynamics; state, step, extra))
end
function transition_logdensity(dynamics::LatentDynamics; next_state, state, step, extra)
    check_transition_distribution(dynamics)
    return logpdf(transition_distribution(dynamics; state, step, extra), next_state)
end

function check_observation_distribution(process::ObservationProcess)
    if !hasmethod(
        observation_distribution,
        Tuple{typeof(process),AbstractVector,AbstractVector,Integer},
    )
        calling_func = StackTraces.stacktrace()[2].func
        error("neither $calling_func nor observation_distribution is defined for $process")
    end
end
function observation(rng::AbstractRNG, process::ObservationProcess; state, step, extra)
    check_observation_distribution(process)
    return rand(rng, observation_distribution(process; state, step, extra))
end
function observation_logdensity(
    process::ObservationProcess; observation, state, step, extra
)
    check_observation_distribution(process)
    return logpdf(observation_distribution(process; state, step, extra), observation)
end
