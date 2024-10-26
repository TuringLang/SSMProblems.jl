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

function sample(
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

    # Backward sampling - exact
    idx_ref = rand(rng, Categorical(weights(particles.filtered)), M)
    trajectories = Array{eltype(model.dyn)}(undef, n_timestep, M)

    trajectories[end, :] = particles.filtered[idx_ref]
    for step in (n_timestep - 1):-1:1
        for j in 1:M
            backward_weights = backward(
                model::StateSpaceModel,
                step,
                trajectories[step + 1],
                recorder.particles[step, :],
                recorder.log_weights[step, :];
                kwargs...,
            )
            ancestor = rand(rng, Categorical(softmax(backward_weights)))
            trajectories[step, j] = recorder.particles[step, ancestor]
        end
    end
    return trajectories
end

function backward(
    model::StateSpaceModel, step::Integer, state, particles::T, log_weights::WT; kwargs...
) where {T,WT}
    transitions = map(
        x -> SSMProblems.logdensity(model.dyn, step, x, state; kwargs...), particles
    )
    return log_weights + transitions
end
