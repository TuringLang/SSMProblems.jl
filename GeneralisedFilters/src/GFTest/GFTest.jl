module GFTest

using CUDA
using Distributions
using LinearAlgebra
using PDMats
using Random
using StaticArrays

using GeneralisedFilters
using SSMProblems

include("utils.jl")
include("models/linear_gaussian.jl")
include("models/dummy_linear_gaussian.jl")
include("models/dummy_discrete.jl")
include("models/mixture_observation.jl")
include("proposals.jl")
include("resamplers.jl")

export MixtureObservation

end
