using Random
using SSMProblems
using Distributions
using Plots
using StatsFuns

# Particle Filter 
abstract type Node{T} end

struct Root{T} <: Node{T} end
Root(T) = Root{T}()
Root() = Root(Any)

struct Particle{T} <: Node{T}
    parent::Node{T}
    state::T
end

Particle(state::T) where {T} = Particle(Root(T), state)

Base.show(io::IO, p::Particle{T}) where {T} = print(io, "Particle{$T}($(p.state))")

const ParticleContainer{T} = AbstractVector{<:Particle{T}}

"""
    linearize(particle)

Return the trace of a particle, i.e. the sequence of states from the root to the particle.
"""
function linearize(particle::Particle{T}) where {T}
    trace = T[]
    current = particle
    while !isa(current, Root)
        push!(trace, current.state)
        current = current.parent
    end
    return trace
end

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
    particles::ParticleContainer,
    observations::AbstractArray,
    resampling=systematic_resampling,
    threshold=0.5,
)
    N = length(particles)
    logweights = zeros(N)

    for (timestep, observation) in enumerate(observations)
        # Resample step
        weights = get_weights(logweights)
        if ess(weights) <= threshold * N
            idx = resampling(rng, weights)
            particles = particles[idx]
            fill!(logweights, 0)
        end

        # Mutation step
        for i in eachindex(particles)
            latent_state = transition!!(rng, model, timestep, particles[i].state)
            particles[i] = Particle(particles[i], latent_state)
            logweights[i] += emission_logdensity(
                model, timestep, particles[i].state, observation
            )
        end
    end

    # Return unweighted set
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
        Particle(state)
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
T = 250
seed = 1
N = 1_000
rng = MersenneTwister(seed)

model = LinearSSM(0.2, 0.7)

f0() = Normal(0, 1)
f(t::Int, x::Float64) = Normal(x, model.v)
g(t::Int, y::Float64) = Normal(y, model.u)

# Generate synthtetic data
x, observations = zeros(T), zeros(T)
x[1] = rand(rng, f0())
for t in 1:T
    observations[t] = rand(rng, g(t, x[t]))
    if t < T
        x[t + 1] = rand(rng, f(t, x[t]))
    end
end

# Model dynamics
function transition!!(rng::AbstractRNG, model::LinearSSM)
    return rand(rng, f0())
end

function transition!!(rng::AbstractRNG, model::LinearSSM, timestep::Int, state::Float64)
    return rand(rng, f(timestep, state))
end

function emission_logdensity(
    model::LinearSSM, timestep::Int, state::Float64, observation::Float64
)
    return logpdf(g(timestep, state), observation)
end

# Sample latent state trajectories
samples = sample(rng, LinearSSM(), N, observations)
traces = reverse(hcat(map(linearize, samples)...))

scatter(traces; color=:black, opacity=0.3, label=false)
plot!(x; label="True state")
plot!(mean(traces; dims=2); label="Posterior mean")
