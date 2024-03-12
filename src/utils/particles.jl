"""
    Common concrete implementations of Particle types for Particle Filter kernels.
"""
module Utils

abstract type Node{T} end

"""
    Particle{T}

Particle as immutable LinkedList. 
"""
mutable struct Particle{T} <: Node{T}
    state::T
    parent::Node{T}
    child::Node{T}

    function Particle{T}() where {T}
        particle = new{T}()
        particle.parent = particle
        particle.child = particle
        return particle
    end

    function Particle(state::T) where {T}
        particle = new{T}(state)
        return particle
    end
end

function Particle(state::T, parent::Particle{T}) where {T}
    particle = Particle(state)
    setfield!(particle, :parent, parent)
    setfield!(parent, :child, particle)
    return particle
end

Base.show(io::IO, p::Particle{T}) where {T} = print(io, "Particle{$T}($(p.state))")

const ParticleContainer{T} = AbstractVector{<:Particle{T}}

"""
    linearize(particle)

Return the trace of a particle, i.e. the sequence of states from the root to the particle.
"""
function linearize(particle::Particle{T}) where {T}
    trace = T[]
    current = particle
    while isdefined(current, :parent)
        push!(trace, current.state)
        current = current.parent
    end
    push!(trace, current.state)
    return trace
end

function rewind(particle::Particle)
    current = particle
    while isdefined(current, :parent)
        current = current.parent
    end
    return current
end

end
