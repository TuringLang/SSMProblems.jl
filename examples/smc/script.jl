# # Partilce Filter with adaptive resampling
using Random
using SSMProblems
using Distributions
using Plots
using StatsFuns

# Particle Filter 
ess(weights) = inv(sum(abs2, weights))
get_weights(logweights::T) where {T<:AbstractVector{<:Real}} = StatsFuns.softmax(logweights)

function systematic_resampling(
    rng::AbstractRNG, weights::AbstractVector{<:Real}, n::Integer=length(weights)
)
    return rand(rng, Distributions.Categorical(weights), n)
end

function sweep!(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    particles::SSMProblems.Utils.ParticleContainer,
    observations::AbstractArray,
    resampling=systematic_resampling,
    threshold=0.5,
)
    N = length(particles)
    logweights = zeros(N)

    for (timestep, observation) in enumerate(observations)
        weights = get_weights(logweights)
        if ess(weights) <= threshold * N
            idx = resampling(rng, weights)
            particles = particles[idx]
            fill!(logweights, 0)
        end

        for i in eachindex(particles)
            latent_state = transition!!(rng, model, particles[i].state, timestep)
            particles[i] = SSMProblems.Utils.Particle(particles[i], latent_state)
            logweights[i] += emission_logdensity(
                model, particles[i].state, observation, timestep
            )
        end
    end

    idx = resampling(rng, get_weights(logweights))
    return particles[idx]
end

# Turing style sample method
function sample(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    n::Int,
    observations::AbstractVector;
    resampling=systematic_resampling,
    threshold=0.5,
)
    particles = map(1:N) do i
        state = transition!!(rng, model)
        SSMProblems.Utils.Particle(state)
    end
    samples = sweep!(rng, model, particles, observations, resampling, threshold)
    return samples
end

# Inference code
Base.@kwdef struct LinearSSM <: AbstractStateSpaceModel
    v::Float64 = 0.2 # Transition noise stdev
    u::Float64 = 0.7 # Observation noise stdev
end

# Simulation
T = 150
seed = 1
N = 500
rng = MersenneTwister(seed)

model = LinearSSM(0.2, 0.7)

f0(::LinearSSM) = Normal(0, 1)
f(x::Float64, model::LinearSSM) = Normal(x, model.v)
g(y::Float64, model::LinearSSM) = Normal(y, model.u)

# Generate synthtetic data
x, observations = zeros(T), zeros(T)
x[1] = rand(rng, f0(model))
for t in 1:T
    observations[t] = rand(rng, g(x[t], model))
    if t < T
        x[t + 1] = rand(rng, f(x[t], model))
    end
end

# Model dynamics
function transition!!(rng::AbstractRNG, model::LinearSSM)
    return rand(rng, f0(model))
end

function transition!!(rng::AbstractRNG, model::LinearSSM, state::Float64, ::Int)
    return rand(rng, f(state, model))
end

function emission_logdensity(model::LinearSSM, state::Float64, observation::Float64, ::Int)
    return logpdf(g(state, model), observation)
end

# Sample latent state trajectories
samples = sample(rng, LinearSSM(), N, observations)
traces = reverse(hcat(map(SSMProblems.Utils.linearize, samples)...))

scatter(traces[:, 1:10]; color=:black, opacity=0.7, label=false)
plot!(x; label="True state", linewidth=2)
plot!(mean(traces; dims=2); label="Posterior mean", color=:orange, linewidth=2)
