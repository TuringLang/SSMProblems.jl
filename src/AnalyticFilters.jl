module AnalyticFilters

using AbstractMCMC: AbstractMCMC
import Distributions: MvNormal
import Random: AbstractRNG, default_rng
using SSMProblems

abstract type FilteringAlgorithm end

# Model types
include("models/linear_gaussian.jl")
include("models/hierarchical.jl")

# Filtering/smoothing algorithms
include("algorithms/kalman.jl")
include("algorithms/rbpf.jl")

end
