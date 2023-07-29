using Random
using SSMProblems
using Distributions
using Plots
using StatsFuns

# Particle Filter implementation
struct Particle{T<:AbstractStateSpaceModel,V} <: AbstractParticle{T}
    parent::Union{Particle,Nothing}
    state::V
end

function Particle(parent, state, model::T) where {T<:AbstractStateSpaceModel}
    return Particle{T, particleof(model)}(parent, state)
end
function Particle(model::T) where {T<:AbstractStateSpaceModel}
    N = dimension(model)
    V = particleof(model)
    state = N == 1 ? zero(V) : zeros(V, N)
    return Particle{T, V}(nothing, state)
end
Base.show(io::IO, p::Particle) = print(io, "Particle($(p.state))")

"""
    linearize(particle)

Return the trace of a particle, i.e. the sequence of states from the root to the particle.
"""
function linearize(particle::Particle{T, V}) where {T, V}
    trace = V[]
    current = particle
    while !isnothing(current)
        push!(trace, current.state)
        current = current.parent
    end
    return trace[1:(end - 1)]
end

const ParticleContainer{T} = AbstractVector{<:Particle{T}}

# Specialize `isdone` to the concrete `Particle` type
function isdone(t, model::AbstractStateSpaceModel, particles::ParticleContainer)
    return all(map(particle -> isdone(t, model, particle), particles))
end

ess(weights) = inv(sum(abs2, weights))
get_weights(logweights::T) where {T<:AbstractVector{<:Real}} = StatsFuns.softmax(logweights)

function systematic_resampling(
    rng::AbstractRNG, weights::AbstractVector{<:Real}, n::Integer=length(weights)
)
    return rand(rng, Distributions.Categorical(weights), n)
end

function sweep!(rng::AbstractRNG, model::AbstractStateSpaceModel, particles::ParticleContainer, resampling, threshold=0.5)
    t = 1
    N = length(particles)
    logweights = zeros(length(particles))
    while !isdone(t, model, particles)

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
            mutated = transition!!(rng, t, model, parent)
            particles[i] = mutated
            logweights[i] += emission_logdensity(t, model, particles[i])
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
    samples = sweep!(rng, model, particles, resampling, threshold)
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

f0(t) = Normal(0, 1)
f(t, x) = Normal(x, model.v)
g(t, x) = Normal(x, model.u)

# Generate synthtetic data
x, observations = zeros(T), zeros(T)
x[1] = rand(rng, f0(1))
for t in 1:T
    observations[t] = rand(rng, g(t, x[t]))
    if t < T
        x[t + 1] = rand(rng, f(t, x[t]))
    end
end

function transition!!(
    rng::AbstractRNG, t::Int, model::LinearSSM, particle::AbstractParticle
)
    if t == 1
        return Particle(particle, rand(rng, f0(t)), model)
    else
        return Particle(particle, rand(rng, f(t, particle.state)), model)
    end
end

function emission_logdensity(t, model::LinearSSM, particle::AbstractParticle)
    return logpdf(g(t, particle.state), observations[t])
end

# isdone
isdone(t, ::LinearSSM, ::AbstractParticle) = t > T

# Type of latent space
# particleof(::S) :: S -> T
# f(t, x) : Int -> T -> T
particleof(::LinearSSM) = Float64
dimension(::LinearSSM) = 1

samples = sample(rng, N, LinearSSM())
traces = reverse(hcat(map(linearize, samples)...))

#scatter(traces; color=:black, opacity=0.3, label=false)
plot(x; label="True state")
plot!(mean(traces, dims=2); label="Posterior mean")

gui()
