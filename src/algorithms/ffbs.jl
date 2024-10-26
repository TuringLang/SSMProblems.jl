export FFBS

abstract type AbstractSmoother <: AbstractSampler end

struct FFBS{T<:AbstractParticleFilter}
    filter::T
end

""" 
    smooth(rng::AbstractRNG, alg::AbstractSmooterh, model::AbstractStateSpaceModel, obs::AbstractVector, M::Integer; callback, kwargs...)
"""
function smooth end

struct WeightedParticleRecorderCallback{T,WT}
    particles::Array{T}
    log_weights::Array{WT}
end

function (callback::WeightedParticleRecorderCallback)(
    model, filter, step, states, data; kwargs...
)
    filtered_states = states.filtered
    callback.particles[step, :] = filtered_states.particles
    callback.log_weights[step, :] = filtered_states.log_weights
    return nothing
end

function smooth(
    rng::Random.AbstractRNG,
    model::StateSpaceModel{T,LDT},
    alg::FFBS{<:BootstrapFilter{N}},
    obs::AbstractVector,
    M::Integer;
    callback=nothing,
    kwargs...,
) where {T,LDT,N}
    n_timestep = length(obs)
    recorder = WeightedParticleRecorderCallback(
        Array{eltype(model.dyn)}(undef, n_timestep, N), Array{T}(undef, n_timestep, N)
    )

    particles, _ = filter(rng, model, alg.filter, obs; callback=recorder, kwargs...)
    idx_ref = rand(rng, Categorical(weights(particles.filtered)), M)
    trajectories = Array{eltype(model.dyn)}(undef, n_timestep, M)

    forward_state = particles.filtered[idx_ref]
    trajectories[end, :] = forward_state
    for step in (n_timestep - 1):-1:1
        for j in 1:M
            transitions = map(
                x ->
                    SSMProblems.logdensity(model.dyn, step, forward_state[j], x; kwargs...),
                recorder.particles[step, :],
            )
            backward_weights = recorder.log_weights[step, :] + transitions
            ancestor = rand(rng, Categorical(softmax(backward_weights)))
            trajectories[step, j] = recorder.particles[step, ancestor]
        end
    end
    return trajectories
end
