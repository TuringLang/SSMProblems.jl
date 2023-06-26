using Random
using SSMProblems
using Distributions
using Plots
using StatsFuns

# Particle Filter implementation
struct Particle{T} # Here we just need a tree
    parent::Union{Particle,Nothing}
    state::T
end

Particle(state::T) where {T} = Particle(nothing, state)
Particle() = Particle(nothing, nothing)
Base.show(io::IO, p::Particle) = print(io, "Particle($(p.state))")

"""
    linearize(particle)

Return the trace of a particle, i.e. the sequence of states from the root to the particle.
"""
function linearize(particle::Particle{T}) where {T}
    trace = T[]
    parent = particle.parent
    while !isnothing(parent)
        push!(trace, parent.state)
        parent = parent.parent
    end
    return trace
end

ParticleContainer = AbstractVector{<:Particle}

ess(weights) = inv(sum(abs2, weights))
get_weights(logweights::T) where {T<:AbstractVector{<:Real}} = StatsFuns.softmax(logweights)

function systematic_resampling(
    rng::AbstractRNG, weights::AbstractVector{<:Real}, n::Integer=length(weights)
)
    return rand(rng, Distributions.Categorical(weights), n)
end

function sweep!(rng::AbstractRNG, particles::ParticleContainer, resampling, threshold=0.5)
    t = 1
    N = length(particles)
    logweights = zeros(length(particles))
    while !isdone(t, particles[1].state)

        # Resample step
        weights = get_weights(logweights)
        if ess(weights) <= threshold * N
            idx = resampling(rng, weights)
            particles = particles[idx]
            logweights = zeros(length(particles))
        end

        # Mutation step
        for i in eachindex(particles)
            parent = particles[i]
            mutated = transition!!(rng, t, parent.state)
            particles[i] = Particle(parent, mutated)
            logweights[i] += emission_logdensity(t, particles[i].state)
        end

        t += 1
    end

    # Return unweighted set
    idx = resampling(rng, get_weights(logweights))
    return particles[idx]
end

function sweep!(rng::AbstractRNG, n::Int, resampling, threshold=0.5)
    particles = [Particle(0.0) for _ in 1:n]
    return sweep!(rng, particles, resampling, threshold)
end

# Inference code
Base.@kwdef struct Parameters
    v::Float64 = 0.2 # Transition noise stdev
    u::Float64 = 0.7 # Observation noise stdev
end

# Simulation
T = 250
seed = 1
N = 1000
rng = MersenneTwister(seed)
params = Parameters(; v=0.2, u=0.7)

function transition!!(rng::AbstractRNG, t::Int, state=nothing)
    if isnothing(state)
        return rand(rng, Normal(0, 1))
    end
    return rand(rng, Normal(state, params.v))
end

function emission_logdensity(t, state)
    return logpdf(Normal(state, params.u), observations[t])
end

isdone(t, state) = t > T
isdone(t, ::Nothing) = false

x, observations = zeros(T), zeros(T)
x[1] = rand(rng, Normal(0, 1))
for t in 1:T
    observations[t] = rand(rng, Normal(x[t], params.u))
    if t < T
        x[t + 1] = rand(rng, Normal(x[t], params.v))
    end
end

samples = sweep!(rng, fill(Particle(x[1]), N), systematic_resampling)
traces = reverse(hcat(map(linearize, samples)...))

scatter(traces; color=:black, opacity=0.3, label=false)
plot!(x; label="True state")
