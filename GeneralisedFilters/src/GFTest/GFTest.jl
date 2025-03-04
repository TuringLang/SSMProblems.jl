module GFTest

using CUDA
using LinearAlgebra
using NNlib
using Random

using GeneralisedFilters
using SSMProblems

include("utils.jl")
include("models/linear_gaussian.jl")
include("models/dummy_linear_gaussian.jl")

end
