using Random
using SSMProblems
using Distributions
using Plots
using StatsFuns

# Particle Filter implementation
mutable struct Particle{T<:AbstractStateSpaceModel} <: AbstractParticle{T}
    parent::Union{Particle,Nothing}
    model::T
end

Particle(model::T) where {T} = Particle(nothing, model)
Particle() = Particle(nothing, nothing)
Base.show(io::IO, p::Particle) = print(io, "Particle($(p.model))")

function set_parent!(particle::Particle, parent::Particle)
    setproperty!(particle, :parent, parent)
    return particle
end

"""
    linearize(particle)

Return the trace of a particle, i.e. the sequence of states from the root to the particle.
"""
function linearize(particle::Particle{T}) where {T}
    trace = T[]
    parent = particle
    while !isnothing(parent)
        push!(trace, parent.model)
        parent = parent.parent
    end
    return trace[1:(end - 1)]
end

const ParticleContainer = AbstractVector{<:Particle}

# Specialize `isdone` to the concrete `Particle` type
function isdone(t, particles::ParticleContainer)
    return all(map(particle -> isdone(t, particle), particles))
end

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
    while !isdone(t, particles)

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
            mutated = transition!!(rng, t, parent)
            mutated = set_parent!(mutated, parent)
            particles[i] = mutated
            logweights[i] += emission_logdensity(t, particles[i].model)
        end

        t += 1
    end

    # Return unweighted set
    idx = resampling(rng, get_weights(logweights))
    return particles[idx]
end

function sample(
    rng::AbstractRNG,
    n::Int,
    model::AbstractStateSpaceModel;
    resampling=systematic_resampling,
    threshold=0.5,
)
    particles = fill(Particle(model), N)
    samples = sweep!(rng, particles, resampling, threshold)
    return samples
end

# Inference code
Base.@kwdef struct Parameters
    v::Float64 = 0.2 # Transition noise stdev
    u::Float64 = 0.7 # Observation noise stdev
end

struct LinearSSM{T} <: AbstractStateSpaceModel
    state::T
end

LinearSSM() = LinearSSM(zero(Float64))
dimension(::LinearSSM) = 1

# Simulation
T = 250
seed = 1
N = 1_000
rng = MersenneTwister(seed)
params = Parameters(; v=0.2, u=0.7)

f0(t) = Normal(0, 1)
f(t, x) = Normal(x, params.v)
g(t, x) = Normal(x, params.u)

# Generate synthtetic data
x, observations = zeros(T), zeros(T)
x[1] = rand(rng, f0(1))
for t in 1:T
    observations[t] = rand(rng, g(t, x[t]))
    if t < T
        x[t + 1] = rand(rng, f(t, x[t]))
    end
end

function transition!!(rng::AbstractRNG, t::Int, particle::Particle{<:LinearSSM})
    if t == 1
        return Particle(LinearSSM(rand(rng, f0(t))))
    else
        return Particle(LinearSSM(rand(rng, f(t, particle.model.state))))
    end
end

function emission_logdensity(t, model::LinearSSM)
    return logpdf(g(t, model.state), observations[t])
end

isdone(t, ::Particle{LinearSSM{F}}) where {F} = t > T

samples = sample(rng, N, LinearSSM())
traces = map(model -> model.state, reverse(hcat(map(linearize, samples)...)))

scatter(traces; color=:black, opacity=0.3, label=false)
plot!(x; label="True state")
