module GFTest

using CUDA
using LinearAlgebra
using PDMats
using Random
using StaticArrays

using GeneralisedFilters
using SSMProblems

export check_mc_estimate, weighted_stats

include("utils.jl")
include("models/linear_gaussian.jl")
include("models/dummy_linear_gaussian.jl")
include("proposals.jl")
include("resamplers.jl")
include("ci_testing.jl")

end
