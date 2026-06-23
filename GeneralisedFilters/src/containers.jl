using LogExpFunctions
using DataStructures: Stack
using Random: rand

export AbstractLikelihood, InformationLikelihood, DiscreteLikelihood, log_likelihoods
export ReferenceTrajectory
export DenseParticleContainer, ParticleTree

"""Containers used for storing representations of the filtering distribution."""

## TYPELESS INITIALIZERS ###################################################################

"""
    TypelessZero

A lazy promotion for uninitialized particle weights whos type is not yet known at the first
simulation of a particle filter.
"""
struct TypelessZero <: Number end

Base.convert(::Type{T}, ::TypelessZero) where {T<:Number} = zero(T)
Base.convert(::Type{TypelessZero}, ::TypelessZero) = TypelessZero()

Base.:+(::TypelessZero, ::TypelessZero) = TypelessZero()

Base.promote_rule(::Type{TypelessZero}, ::Type{T}) where {T<:Number} = T
Base.promote_rule(::Type{TypelessZero}, ::Type{TypelessZero}) = TypelessZero

Base.zero(::TypelessZero) = TypelessZero()
Base.zero(::Type{TypelessZero}) = TypelessZero()

Base.iszero(::TypelessZero) = true
Base.isone(::TypelessZero) = false

Base.show(io::IO, ::TypelessZero) = print(io, "TypelessZero()")

"""
    TypelessBaseline

A lazy promotion for the computation of log-likelihood baslines given a collection of
unweighted particles.
"""
struct TypelessBaseline <: Number
    N::Int64
end

# Constructors for compatibility with Base.Number
TypelessBaseline(x::TypelessBaseline) = x
TypelessBaseline(x::Base.TwicePrecision) = TypelessBaseline(Int64(x))
TypelessBaseline(x::AbstractChar) = TypelessBaseline(Int64(x))

Base.convert(::Type{T}, b::TypelessBaseline) where {T<:Number} = T(log(b.N))
Base.promote_rule(::Type{TypelessBaseline}, ::Type{T}) where {T<:Number} = T

Base.iszero(::TypelessBaseline) = false
Base.isone(::TypelessBaseline) = false

function LogExpFunctions.logsumexp(weights::AbstractVector{TypelessZero})
    return TypelessBaseline(length(weights))
end

function LogExpFunctions.softmax(x::AbstractVector{TypelessZero})
    # TODO: horrible, but theoretically never used... except in the unit tests
    return fill(1 / length(x), length(x))
end

Base.:+(::TypelessZero, b::TypelessBaseline) = b
Base.:+(b::TypelessBaseline, ::TypelessZero) = b

Base.show(io::IO, b::TypelessBaseline) = print(io, "Typeless(log($(b.N)))")

## PARTICLES ###############################################################################

"""
    Particle

A container representing a single particle in a particle filter distribution, composed of a
weighted sampled (stored as a log weight) and its ancestor index.
"""
struct Particle{ST,WT,AT<:Integer}
    state::ST
    log_w::WT
    ancestor::AT
end

# NOTE: this is only ever used for initializing a particle filter
const UnweightedParticle{ST,AT} = Particle{ST,TypelessZero,AT}

Particle(state, ancestor) = Particle(state, TypelessZero(), ancestor)
Particle(particle::UnweightedParticle, ancestor) = Particle(particle.state, ancestor)
function Particle(particle::Particle{<:Any,WT}, ancestor) where {WT<:Real}
    return Particle(particle.state, zero(WT), ancestor)
end

log_weight(p::Particle) = p.log_w

"""
    RBState

A container representing a single state with a Rao-Blackwellised component. This differs
from a `HierarchicalState` which contains a sample of the conditionally analytical state
rather than the distribution itself.

# Fields
- `x::XT`: The sampled state component
- `z::ZT`: The Rao-Blackwellised distribution component
"""
struct RBState{XT,ZT}
    x::XT
    z::ZT
end

"""
    ParticleDistribution

A container for particle filters which composes a collection of weighted particles (with
their ancestories) into a distibution-like object.

# Fields
- `particles::VT`: Vector of weighted particles
- `ll_baseline::WT`: Baseline for computing log-likelihood increment. A scalar that caches
  the unnormalized logsumexp of weights before update (for standard PF/guided filters)
  or a modified value that includes APF first-stage correction (for auxiliary PF).
"""
mutable struct ParticleDistribution{WT,PT<:Particle,VT<:AbstractVector{PT}}
    particles::VT
    ll_baseline::WT
end

# Helper functions to make ParticleDistribution behave like a collection
Base.collect(state::ParticleDistribution) = state.particles
Base.length(state::ParticleDistribution) = length(state.particles)
Base.keys(state::ParticleDistribution) = LinearIndices(state.particles)
Base.iterate(state::ParticleDistribution, i) = iterate(state.particles, i)
Base.iterate(state::ParticleDistribution) = iterate(state.particles)

# Not sure if this is kosher, since it doesn't follow the convention of Base.getindex
Base.@propagate_inbounds Base.getindex(state::ParticleDistribution, i) = state.particles[i]

log_weights(state::ParticleDistribution) = map(p -> log_weight(p), state.particles)
get_weights(state::ParticleDistribution) = softmax(log_weights(state))

# Helpers for StatsBase compatibility
StatsBase.weights(state::ParticleDistribution) = StatsBase.Weights(get_weights(state))

"""
    marginalise!(state::ParticleDistribution)

Compute the log-likelihood increment and normalize particle weights. This function:
1. Computes LSE of current (post-observation) log-weights
2. Calculates ll_increment = LSE_after - ll_baseline
3. Normalizes weights by subtracting LSE_after
4. Resets ll_baseline to 0.0

The ll_baseline field handles both standard particle filter and auxiliary particle filter
cases through a single-scalar caching mechanism. For standard PF, ll_baseline equals the
LSE before adding observation weights. For APF with resampling, it includes first-stage
correction terms computed during the APF resampling step.
"""
function marginalise!(state::ParticleDistribution, particles)
    # Compute logsumexp after adding observation likelihoods
    LSE_after = logsumexp(log_weight.(particles))

    # Compute log-likelihood increment: works for both PF and APF cases
    ll_increment = LSE_after - state.ll_baseline

    # Create new particles with normalized weights
    particles = map(p -> Particle(p.state, p.log_w - LSE_after, p.ancestor), particles)

    # Reset baseline for next iteration
    new_state = ParticleDistribution(particles, zero(ll_increment))
    return new_state, ll_increment
end

## LIKELIHOOD CONTAINERS ###################################################################

"""
    AbstractLikelihood

Abstract type for backward likelihood representations used in smoothing and ancestor sampling.

Subtypes represent the predictive likelihood p(y | x) in different forms depending
on the state space structure (continuous Gaussian vs discrete).
"""
abstract type AbstractLikelihood end

"""
    InformationLikelihood <: AbstractLikelihood

A container representing an unnormalized Gaussian likelihood p(y | x) in information form,
parameterized by natural parameters (λ, Ω).

The unnormalized log-likelihood is given by:
    log p(y | x) ∝ λ'x - (1/2)x'Ωx

This representation is particularly useful in backward filtering algorithms and
Rao-Blackwellised particle filtering, where it represents the predictive likelihood
p(y_{t:T} | x_t) conditioned on future observations.

# Fields
- `λ::λT`: The natural parameter vector (information vector)
- `Ω::ΩT`: The natural parameter matrix (information/precision matrix)

# See also
- [`natural_params`](@ref): Extract the natural parameters (λ, Ω)
- [`BackwardInformationPredictor`](@ref): Algorithm that uses this representation
"""
struct InformationLikelihood{λT,ΩT} <: AbstractLikelihood
    λ::λT
    Ω::ΩT
end

"""
    natural_params(state::InformationLikelihood)

Extract the natural parameters (λ, Ω) from an InformationLikelihood.

Returns a tuple `(λ, Ω)` where λ is the information vector and Ω is the
information/precision matrix.
"""
function natural_params(state::InformationLikelihood)
    return state.λ, state.Ω
end

"""
    DiscreteLikelihood <: AbstractLikelihood

A container representing the backward likelihood β_t(i) = p(y | x = i) for discrete
state spaces, stored in log-space for numerical stability.

This representation is used in backward filtering algorithms for discrete SSMs (HMMs) and
Rao-Blackwellised particle filtering with discrete inner states.

# Fields
- `log_β::VT`: Vector of log backward likelihoods, where `log_β[i] = log p(y | x = i)`

# See also
- [`BackwardDiscretePredictor`](@ref): Algorithm that uses this representation
"""
struct DiscreteLikelihood{VT<:AbstractVector} <: AbstractLikelihood
    log_β::VT
end

"""
    log_likelihoods(state::DiscreteLikelihood)

Extract the log backward likelihoods from a DiscreteLikelihood.
"""
log_likelihoods(state::DiscreteLikelihood) = state.log_β

## REFERENCE TRAJECTORY ###################################################################

"""
    ReferenceTrajectory{T0, T, VT}

A 0-indexed trajectory container where the initial state (`x0`) may have a different type
than the subsequent states (`xs`). This is common in Rao-Blackwellised settings where the
prior's inner distribution differs from the filtered distributions at subsequent time steps.

When `T0 == T`, the element type collapses to a concrete `T` (since `Union{T,T} == T`).

# Fields
- `x0::T0`: State at time 0 (typically from the prior)
- `xs::VT`: States at times 1, 2, …, T (an `AbstractVector{T}`)

Indices are 0-based: `traj[0] === x0`, `traj[t] === xs[t]` for t ≥ 1.
"""
struct ReferenceTrajectory{T0,T,VT<:AbstractVector{T}} <: AbstractVector{Union{T0,T}}
    x0::T0
    xs::VT
end

Base.size(r::ReferenceTrajectory) = (length(r.xs) + 1,)
Base.axes(r::ReferenceTrajectory) = (0:length(r.xs),)
Base.IndexStyle(::Type{<:ReferenceTrajectory}) = IndexLinear()

Base.@propagate_inbounds function Base.getindex(r::ReferenceTrajectory, i::Integer)
    return i == 0 ? r.x0 : r.xs[i]
end

Base.map(f, r::ReferenceTrajectory) = ReferenceTrajectory(f(r.x0), map(f, r.xs))

function Base.:(==)(a::ReferenceTrajectory, b::ReferenceTrajectory)
    return a.x0 == b.x0 && a.xs == b.xs
end

function Base.show(io::IO, r::ReferenceTrajectory)
    print(io, "ReferenceTrajectory(x0=", r.x0, ", xs=", r.xs, ")")
    return nothing
end
Base.show(io::IO, ::MIME"text/plain", r::ReferenceTrajectory) = show(io, r)

## DENSE PARTICLE STORAGE #################################################################

"""
    DenseParticleContainer{T0, T, WT}

Full-history particle storage: the initial particle states (time 0), plus subsequent
particle states, weights, and ancestor indices for times 1, 2, …, T.

`T0` is the type of the initial particle states; `T` is the type of the subsequent
particle states. When they match, use the one-argument constructor — pushing subsequent
states with a different type will raise an error.

# Fields
- `initial_states::Vector{T0}`: Time-0 particle states (fixed length N)
- `states::Vector{Vector{T}}`: Particle states at times 1..T
- `weights::Vector{Vector{WT}}`: Log weights at times 1..T
- `ancestors::Vector{Vector{Int}}`: Ancestor indices at times 1..T

# Constructors

    DenseParticleContainer(initial_states)
    DenseParticleContainer(initial_states, T)
    DenseParticleContainer(initial_states, states_t1, weights_t1, ancestors_t1)

- First form: assumes `T == T0`.
- Second form: explicit subsequent-state type `T` (the container starts empty for t≥1).
- Third form: construct directly from the initial states and the time-1 states, weights,
  and ancestors — `T` is inferred from `states_t1`. Prefer this when the initial and
  subsequent state types differ so that `T` is inferred from concrete data.

# Mutation

Use `push!(container, states, weights, ancestors)` to append a new time step. Passing
a `states` vector whose eltype does not match `T` raises an `ArgumentError`.
"""
struct DenseParticleContainer{T0,T,WT}
    initial_states::Vector{T0}
    states::Vector{Vector{T}}
    weights::Vector{Vector{WT}}
    ancestors::Vector{Vector{Int}}
end

function DenseParticleContainer(initial_states::Vector{T0}) where {T0}
    return DenseParticleContainer{T0,T0,Float64}(
        initial_states,
        Vector{Vector{T0}}(),
        Vector{Vector{Float64}}(),
        Vector{Vector{Int}}(),
    )
end

function DenseParticleContainer(initial_states::Vector{T0}, ::Type{T}) where {T0,T}
    return DenseParticleContainer{T0,T,Float64}(
        initial_states,
        Vector{Vector{T}}(),
        Vector{Vector{Float64}}(),
        Vector{Vector{Int}}(),
    )
end

function DenseParticleContainer(
    initial_states::Vector{T0},
    states_t1::Vector{T},
    weights_t1::Vector{WT},
    ancestors_t1::AbstractVector{<:Integer},
) where {T0,T,WT}
    return DenseParticleContainer{T0,T,WT}(
        initial_states, [states_t1], [weights_t1], [Vector{Int}(ancestors_t1)]
    )
end

function Base.push!(
    c::DenseParticleContainer{T0,T,WT},
    states::Vector{T},
    weights::Vector{WT},
    ancestors::AbstractVector{<:Integer},
) where {T0,T,WT}
    push!(c.states, states)
    push!(c.weights, weights)
    push!(c.ancestors, Vector{Int}(ancestors))
    return c
end

function Base.push!(
    c::DenseParticleContainer{T0,T,WT}, states, weights, ancestors
) where {T0,T,WT}
    throw(
        ArgumentError(
            "Subsequent states/weights have type ($(eltype(states)), $(eltype(weights))) " *
            "but the container's subsequent state/weight types are ($T, $WT). If the " *
            "initial and subsequent types are intentionally different, construct the " *
            "container with `DenseParticleContainer(initial_states, states_t1, " *
            "weights_t1, ancestors_t1)` (or `DenseParticleContainer(initial_states, T)`).",
        ),
    )
end

"""
    Particle(container::DenseParticleContainer, t::Integer, i::Integer)

Reconstruct the `Particle` at time `t ≥ 1`, index `i`, from the container's stored
state, weight, and ancestor index.
"""
function Particle(c::DenseParticleContainer, t::Integer, i::Integer)
    return Particle(c.states[t][i], c.weights[t][i], c.ancestors[t][i])
end

"""
    get_ancestry(container::DenseParticleContainer, i::Integer)

Return the trajectory of particle `i` as a [`ReferenceTrajectory`](@ref), walking the
ancestry backwards from the final time to time 0.
"""
function get_ancestry(c::DenseParticleContainer{T0,T}, i::Integer) where {T0,T}
    Tlen = length(c.ancestors)
    if Tlen == 0
        return ReferenceTrajectory(c.initial_states[i], T[])
    end
    xs = Vector{T}(undef, Tlen)
    a = i
    for t in Tlen:-1:1
        xs[t] = c.states[t][a]
        a = c.ancestors[t][a]
    end
    x0 = c.initial_states[a]
    return ReferenceTrajectory(x0, xs)
end

## SPARSE PARTICLE STORAGE ################################################################

function append_to_stack!(s::Stack{T}, a::AbstractVector) where {T}
    for x in a
        push!(s, x)
    end
    return s
end

"""
    ParticleTree{T0, T}

Sparse storage of particle ancestry with separate buffers for initial states (time 0) and
subsequent states (times 1..T). The initial-state buffer is never expanded or pruned; the
subsequent-state buffer grows as needed.

`T0` and `T` are the types of initial and subsequent states respectively.

# Fields
- `initial_states::Vector{T0}`: Time-0 particle states (fixed length N)
- `states::Vector{T}`: Pool for times 1..T (expandable)
- `parents::Vector{Int64}`: For each entry in `states`, the parent's index
- `is_penultimate::Vector{Bool}`: `true` if `parents[i]` indexes `initial_states`,
  `false` if it indexes `states`
- `leaves::Vector{Int64}`: Current leaf indices (into `initial_states` before the first
  insert, into `states` thereafter)
- `offspring::Vector{Int64}`: Offspring counts for `states`
- `free_indices::Stack{Int64}`: Unused slots in `states`
- `leaves_in_initial::Ref{Bool}`: `true` until the first `insert!` call

# Constructors

    ParticleTree(initial_states, M)
    ParticleTree(initial_states, T, M)
    ParticleTree(initial_states, states_t1, ancestors_t1, M)

- First form: assumes `T == T0`.
- Second form: explicit subsequent-state type `T`.
- Third form: construct with time-1 data already inserted — `T` is inferred from
  `states_t1`. Prefer this when initial and subsequent types differ.

# Reference
Jacob, P., Murray L., & Rubenthaler S. (2015). Path storage in the particle filter
[doi:10.1007/s11222-013-9445-x](https://dx.doi.org/10.1007/s11222-013-9445-x)
"""
struct ParticleTree{T0,T}
    initial_states::Vector{T0}
    states::Vector{T}
    parents::Vector{Int64}
    is_penultimate::Vector{Bool}
    leaves::Vector{Int64}
    offspring::Vector{Int64}
    free_indices::Stack{Int64}
    leaves_in_initial::Ref{Bool}
end

function ParticleTree(initial_states::Vector{T0}, M::Integer) where {T0}
    return ParticleTree(initial_states, T0, M)
end

function ParticleTree(initial_states::Vector{T0}, ::Type{T}, M::Integer) where {T0,T}
    states = Vector{T}(undef, M)
    parents = zeros(Int64, M)
    is_penultimate = fill(false, M)
    offspring = zeros(Int64, M)
    free_indices = Stack{Int64}()
    append_to_stack!(free_indices, collect(Int64, M:-1:1))
    leaves = collect(Int64, 1:length(initial_states))
    return ParticleTree{T0,T}(
        initial_states,
        states,
        parents,
        is_penultimate,
        leaves,
        offspring,
        free_indices,
        Ref(true),
    )
end

function ParticleTree(
    initial_states::Vector{T0},
    states_t1::Vector{T},
    ancestors_t1::AbstractVector{<:Integer},
    M::Integer,
) where {T0,T}
    tree = ParticleTree(initial_states, T, M)
    insert!(tree, states_t1, ancestors_t1)
    return tree
end

Base.length(tree::ParticleTree) = length(tree.states)
Base.keys(tree::ParticleTree) = LinearIndices(tree.states)

function prune!(tree::ParticleTree, offspring::AbstractVector{<:Integer})
    if tree.leaves_in_initial[]
        return tree
    end
    setindex!(tree.offspring, offspring, tree.leaves)
    @inbounds for i in eachindex(offspring)
        j = tree.leaves[i]
        while j > 0 && tree.offspring[j] == 0
            push!(tree.free_indices, j)
            if tree.is_penultimate[j]
                break
            end
            parent = tree.parents[j]
            tree.offspring[parent] -= 1
            j = parent
        end
    end
    return tree
end

function Base.insert!(
    tree::ParticleTree{T0,T}, states::Vector{T}, ancestors::AbstractVector{<:Integer}
) where {T0,T}
    parents_of_new = getindex(tree.leaves, ancestors)
    parent_in_initial = tree.leaves_in_initial[]

    if length(tree.free_indices) < length(ancestors)
        @debug "expanding tree"
        expand!(tree)
    end

    @inbounds for i in eachindex(states)
        tree.leaves[i] = pop!(tree.free_indices)
    end
    setindex!(tree.states, states, tree.leaves)
    setindex!(tree.parents, parents_of_new, tree.leaves)
    setindex!(tree.is_penultimate, fill(parent_in_initial, length(states)), tree.leaves)
    tree.leaves_in_initial[] = false
    return tree
end

function Base.insert!(tree::ParticleTree{T0,T}, states, ancestors) where {T0,T}
    throw(
        ArgumentError(
            "Subsequent particle states have type $(eltype(states)) but the tree's " *
            "subsequent-state type is $T. If the initial and subsequent types are " *
            "intentionally different, construct the tree with " *
            "`ParticleTree(initial_states, states_t1, ancestors_t1, M)` (or " *
            "`ParticleTree(initial_states, T, M)`).",
        ),
    )
end

function expand!(tree::ParticleTree)
    M = length(tree)
    resize!(tree.states, 2 * M)
    append!(tree.parents, zeros(Int64, M))
    append!(tree.is_penultimate, falses(M))
    append!(tree.offspring, zeros(Int64, M))
    append_to_stack!(tree.free_indices, collect(Int64, (2 * M):-1:(M + 1)))
    return tree
end

function get_offspring(a::AbstractVector{<:Integer})
    offspring = zero(a)
    for i in a
        offspring[i] += 1
    end
    return offspring
end

"""
    get_ancestry(tree::ParticleTree)

Return all leaf trajectories as a vector of [`ReferenceTrajectory`](@ref), walking from
each leaf up to the corresponding initial state.
"""
function get_ancestry(tree::ParticleTree{T0,T}) where {T0,T}
    N = length(tree.leaves)
    if tree.leaves_in_initial[]
        return [ReferenceTrajectory(tree.initial_states[i], T[]) for i in tree.leaves]
    end
    paths = Vector{ReferenceTrajectory{T0,T,Vector{T}}}(undef, N)
    @inbounds for (k, leaf) in enumerate(tree.leaves)
        xs = T[tree.states[leaf]]
        j = tree.parents[leaf]
        penult = tree.is_penultimate[leaf]
        while !penult
            push!(xs, tree.states[j])
            penult = tree.is_penultimate[j]
            j = tree.parents[j]
        end
        reverse!(xs)
        paths[k] = ReferenceTrajectory(tree.initial_states[j], xs)
    end
    return paths
end

function rand(
    rng::AbstractRNG, tree::ParticleTree{T0,T}, weights::AbstractVector{<:Real}
) where {T0,T}
    b = StatsBase.sample(rng, StatsBase.Weights(weights))
    leaf = tree.leaves[b]
    if tree.leaves_in_initial[]
        return ReferenceTrajectory(tree.initial_states[leaf], T[])
    end
    xs = T[tree.states[leaf]]
    j = tree.parents[leaf]
    penult = tree.is_penultimate[leaf]
    while !penult
        push!(xs, tree.states[j])
        penult = tree.is_penultimate[j]
        j = tree.parents[j]
    end
    reverse!(xs)
    return ReferenceTrajectory(tree.initial_states[j], xs)
end
