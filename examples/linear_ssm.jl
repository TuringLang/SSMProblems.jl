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

function ancestor_logweights(t, logweights::AbstractVector{<:Real}, particles::ParticleContainer, target::Particle)
    return logweights .+ map(particle -> logM(t, particle, target.x), particles)
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
    N = length(particles)
    logweights = zeros(N)
    while !isdone(t, particles[1])

        if ess(weights(logweights)) <= threshold * N
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
    ref = isnothing(reference.child) ? forward!(reference) : reference
    particles[N] = ref
    while !isdone(t, particles[1])

        if ess(weights(logweights)) <= threshold * N
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

function sweep!(rng::AbstractRNG, particles::ParticleContainer, reference::Particle, threshold::Float64=0.5, ancestor=true)
    # Sweep with reference and ancestor resampling
    t = 1
    N = length(particles)
    logweights = zeros(length(particles))
    ref = isnothing(reference.child) ? forward!(reference) : reference
    particles[N] = ref
    while !isdone(t, particles[1])

        idx = resampling(rng, weights(logweights), N-1)
        aidx = t == 1 ? N : ancestor_index(t, logweights, particles, particles[N])
        idx = [idx; aidx]
        logweights = zeros(length(particles))

        particles = particles[idx]
        for n in eachindex(particles)
            if n == N
                ref = ref.child[]
                particles[n] = State(ref.x, Ref(particles[n]), ref.child)# Points to the wrong parent
                # PG: particles[n] = reference.child[]
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

function ancestor_index(t, logweights, particles, target)
    ancestor_weights = weights(ancestor_logweights(t, logweights, particles, target))
    idx = resampling(rng, ancestor_weights, 1)
    return idx
end


# Small helpers, not really needed
#
function replay(particle::State, accessor=ancestor)
    #values = [particle.x]
    values = []
    parent = accessor(particle)
    while !isnothing(parent)
        push!(values, parent.x)
        parent = accessor(parent)
    end
    return values
end

function Base.show(io::IO, part::State)
    buffer = IOBuffer()
    println(buffer, "Particle: $(part.x)")
    println(buffer, "Parents: $(replay(part, ancestor))")
    println(buffer, "Children: $(replay(part, children))")
    print(io, String(take!(buffer)))
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

function logM(t, particle::Particle, x)
    return logpdf(Normal(particle.x, 0.2^2), x)
end

isdone(t, particle::State) = t > T

# Simulation
T = 250
seed = 32
N = 50
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
        println(i)
        particles = sweep!(rng, [State(0.) for _ in 1:N], selected, 0.5)
        selected = particles[rand(1:length(particles))]
        samples[i] = selected
    end
    return samples
end

show(State(0))
samples = pmcmc(rng, M, N)
traces = hcat([replay(samples[i]) for i in 1:M-1]...)
r = sum(abs.(diff(traces; dims=2)) .> 0; dims=2) / M
