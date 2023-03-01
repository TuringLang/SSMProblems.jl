using SSMProblems
using Distributions
using Random
using StatsFuns

mutable struct State{T,V} <: Particle
    x::T
    logw::V # Move away from the particle
    parent::Union{Ref{State{T,V}},Nothing}
    child::Union{Ref{State{T,V}},Nothing}
end

State(x::T, weight::U, parent::V) where {T,U,V<:Particle} = State(x, 0., Ref(parent), nothing)
State(x::T, parent::V) where {T,V<:Particle} = State(x, 0., parent, nothing)
State(x::T) where T = State(x, 0., nothing, nothing)

ParticleContainer = Vector{T} where T <: Particle

function resampling(rng::AbstractRNG, weights::AbstractVector{<:Real}, n::Integer=length(weights))
    return rand(rng, Distributions.sampler(Distributions.Categorical(weights)), n)
end

ess(weights) = inv(sum(abs2, weights))
ess(particles::ParticleContainer) = ess(weights(particles))

weights(particles::ParticleContainer) = StatsFuns.softmax([getfield(particle, :logw) for particle in particles])

# SMC Sweep
function sweep!(rng::AbstractRNG, particles::ParticleContainer, reference::Union{Particle,Nothing}=nothing)
    # Resample particles, mutate, get the new weights
    t = 1
    while !isdone(t, particles[1])
        idx = ess(particles) > 0.5 ? resampling(rng, weights(particles)) : 1:length(particles)
        particles = particles[idx]
        for n in eachindex(particles)
            parent = particles[n]
            mutated = M!!(rng, t, parent)
            particles[n] = State(mutated, parent.logw, parent)
            particles[n].logw += logdensity(t, particles[n])
        end
        t += 1
    end
    return particles
end

# Particle MCMC
function step(rng::AbstractRNG, particles::ParticleContainer, ref::Union{Particle,Nothing})
end

ancestor(particle::State) = isnothing(particle.parent) ? nothing : particle.parent[]
children(particle::State) = isnothing(particle.child) ? nothing : particle.child[]

function replay(particle::State, accessor=ancestor)
    values = [particle.x]
    parent = accessor(particle)
    while !isnothing(parent)
        push!(values, parent.x)
        parent = accessor(parent)
    end
    return values
end

function forward!(particle::State)
    current, prev = particle, ancestor(particle)
    while !isnothing(prev)
        setfield!(prev, :child, Ref(current))
        current, prev = prev, ancestor(prev)
    end
    return current
end

# Model specifics
function M!!(rng::AbstractRNG, t, particle::State)
    if t == 1
        return rand(rng, Normal(0, 1))
    end
    return rand(rng, Normal(particle.x, 0.2^2))
end

function logdensity(t, particle::State)
    return logpdf(Normal(particle.x, 0.2^2), observations[t])
end

isdone(t, particle::State) = t > T

# Simulation
T = 2
seed = 32
N = 3
rng = MersenneTwister(seed)

x = [rand(rng, Normal(0,1))]
observations = [rand(Normal(x[1], 0.2^2))]
for t in 2:T
    push!(x, rand(rng, Normal(x[t-1], 0.2^2)))
    push!(observations, rand(rng, Normal(x[t], 0.2^2)))
end

particles = [State(0.) for _ in 1:N]
particles = sweep!(rng, particles)

new_particles = [State(0.) for _ in 1:N-1]
push!(new_particles, forward!(particles[end]))
particles2 = sweep!(rng, new_particles, new_particles[end])

traces = hcat([replay(particle) for particle in particles]...)
