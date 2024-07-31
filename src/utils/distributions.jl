"""
    Default sampling and log-density methods when corresponding distributions are defined.
"""

function simulate(rng::AbstractRNG, dynamics::LatentDynamics, extra)
    return rand(rng, distribution(dynamics, extra))
end
simulate(dynamics::LatentDynamics, extra) = simulate(default_rng(), dynamics, extra)
function logdensity(dynamics::LatentDynamics, state, extra)
    return logpdf(distribution(dynamics, extra), state)
end

function simulate(rng::AbstractRNG, dynamics::LatentDynamics, state, step, extra)
    return rand(rng, distribution(dynamics, state, step, extra))
end
function simulate(dynamics::LatentDynamics, state, step, extra)
    return simulate(default_rng(), dynamics, state, step, extra)
end
function logdensity(dynamics::LatentDynamics, state, next_state, step, extra)
    return logpdf(distribution(dynamics, state, step, extra), next_state)
end

function simulate(rng::AbstractRNG, process::ObservationProcess, state, step, extra)
    return rand(rng, distribution(process, state, step, extra))
end
function simulate(process::ObservationProcess, state, step, extra)
    return simulate(default_rng(), process, state, step, extra)
end
function logdensity(process::ObservationProcess, state, observation, step, extra)
    return logpdf(distribution(process, state, step, extra), observation)
end
