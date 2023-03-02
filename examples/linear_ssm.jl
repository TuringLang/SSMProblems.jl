using SSMProblems
using Distributions
using Random
using StatsFuns

mutable struct State{T} <: Particle
    x::T
    parent::Union{Ref{State{T}},Nothing} # Should be handled by the sampler ?
    child::Union{Ref{State{T}},Nothing}
end

value(particle::State) = getfield(particle, :x)

State(x::T, parent::V) where {T,V<:Particle} = State(x, Ref(parent), nothing)
State(x::T) where T = State(x, nothing, nothing)

ParticleContainer = Vector{T} where T <: Particle

ancestor(particle::State) = isnothing(particle.parent) ? nothing : particle.parent[]
children(particle::State) = isnothing(particle.child) ? nothing : particle.child[]

function resampling(rng::AbstractRNG, weights::AbstractVector{<:Real}, n::Integer=length(weights))
    return rand(rng, Distributions.sampler(Distributions.Categorical(weights)), n)
end

ess(weights) = inv(sum(abs2, weights))
weights(logweights::T) where {T<:AbstractVector{<:Real}} = StatsFuns.softmax(logweights)

function forward!(particle::State)
    current, prev = particle, ancestor(particle)
    while !isnothing(prev)
        setfield!(prev, :child, Ref(current))
        current, prev = prev, ancestor(prev)
    end
    return current
end

# SMC Sweep
function sweep!(rng::AbstractRNG, particles::ParticleContainer, threshold::Float64=0.5)
    # This seems like a rather generic algorithm
    #
    # idx = resample(logweights)
    # new_particles = particles[idx]
    # new_particles[n] = M!!(t, new_particles[n], ...)
    # logweights[n] = logdensity(t, new_particles[n], ...)
    #
    t = 1
    logweights = zeros(length(particles))
    while !isdone(t, particles[1])

        if ess(weights(logweights)) > threshold
            idx = resampling(rng, weights(logweights))
            logweights = zeros(length(particles))
        else
            idx = 1:length(particles)
        end

        particles = particles[idx]
        for n in eachindex(particles)
            parent = particles[n]
            mutated = M!!(rng, t, parent)
            particles[n] = State(mutated, parent)
            logweights[n] += logdensity(t, particles[n])
        end
        t += 1
    end
    idx = resampling(rng, weights(logweights))
    return particles[idx]
end


function sweep!(rng::AbstractRNG, particles::ParticleContainer, reference::Particle, threshold::Float64=0.5)
    # Sweep with reference
    t = 1
    N = length(particles)
    logweights = zeros(length(particles))
    trace = isnothing(reference.child) ? forward!(reference) : reference
    particles[N] = trace
    while !isdone(t, particles[1])

        if ess(weights(logweights)) > threshold
            idx = resampling(rng, weights(logweights), N-1)
            logweights = zeros(length(particles))
        else
            idx = 1:N-1
        end
        idx = [idx; N]

        particles = particles[idx]
        for n in eachindex(particles)
            if n == N
                particles[n] = particles[n].child[]
            else
                parent = particles[n]
                mutated = M!!(rng, t, parent)
                particles[n] = State(mutated, parent)
            end
            logweights[n] += logdensity(t, particles[n])
        end
        t += 1
    end
    idx = resampling(rng, weights(logweights))
    return particles[idx]
end


# Small helpers, not really needed
#

function replay(particle::State, accessor=ancestor)
    values = [particle.x]
    parent = accessor(particle)
    while !isnothing(parent)
        push!(values, parent.x)
        parent = accessor(parent)
    end
    return values
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
T = 250
seed = 32
N = 100
M = 500
rng = MersenneTwister(seed)

x = [rand(rng, Normal(0,1))]
observations = [rand(Normal(x[1], 0.2^2))]
for t in 2:T
    push!(x, rand(rng, Normal(x[t-1], 0.2^2)))
    push!(observations, rand(rng, Normal(x[t], 0.2^2)))
end

# Particle MCMC
function pmcmc(rng::AbstractRNG, M::Int, N::Int)
    samples = Array{State{Float64}}(undef, M)
    particles = sweep!(rng, [State(0.) for _ in 1:N])
    selected = particles[rand(1:length(particles))]
    samples[1] = selected
    for i in 1:M-1
        particles = sweep!(rng, [State(0.) for _ in 1:N], selected)
        selected = particles[rand(1:length(particles))]
        samples[i] = selected
    end
    return samples
end

samples = pmcmc(rng, M, N)
traces = hcat([replay(samples[i]) for i in 1:M-1]...)

using Plots
scatter!(traces[end-1:-1:1, :], color=:black, label=false, opacity=.4)
plot!(x)
