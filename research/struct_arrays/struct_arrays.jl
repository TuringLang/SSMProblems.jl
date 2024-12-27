using CUDA
using StaticArrays
using StructArrays

struct BatchRaoBlackwellisedParticles{XT,ZT}
    x::XT
    z::ZT
end

# ERROR: CuArray only supports element types that are allocated inline.
# Vector{Float64} is a mutable type
# struct Gaussian{T}
#     μ::Vector{T}
#     Σ::Matrix{T}
# end
struct StaticGaussian{T,D}
    μ::SVector{D,T}
    Σ::SMatrix{D,D,T}
end

gaussians = [StaticGaussian(SVector{2}(rand(2)), SMatrix{2,2}(rand(2, 2))) for _ in 1:3]
gaussians_sa = StructArray(gaussians)

# Even this isn't what we want. This is a vector of vectors. We want an array.
# What about when we convert to CUDA?

gaussians_cuda = replace_storage(CuArray, gaussians_sa)

# Even this doesn't work. What about something even simpler?

struct UnivariateGaussian{T}
    μ::T
    σ::T
end

unigaussians = [UnivariateGaussian(rand(), rand()) for _ in 1:3]
unigaussians_sa = StructArray(unigaussians)

unigaussians_cuda = replace_storage(CuArray, unigaussians_sa)
