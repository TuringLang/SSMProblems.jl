import StaticArrays: SVector, SMatrix

struct BatchedSAVector{T} <: BatchedVector{T}
    data::Array{T,2}
end
Base.eltype(::BatchedSAVector{T}) where {T} = T


