"""
    Common concrete implementations of Particle types for Particle Filter kernels.
"""
module Utils

abstract type Node{T} end

struct Root{T} <: Node{T} end
Root(T) = Root{T}()
Root() = Root(Any)

"""
    Particle{T}

Particle as immutable LinkedList. 
"""
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

end
