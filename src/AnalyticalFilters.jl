module AnalyticalFilters

using AbstractMCMC: AbstractMCMC
import Distributions: MvNormal
import Random: AbstractRNG, default_rng
using GaussianDistributions: pairs, Gaussian
using SSMProblems

include("resamplers.jl")
include("containers.jl")
include("filters.jl")

# Model types
include("models/linear_gaussian.jl")
include("models/discrete.jl")
include("models/hierarchical.jl")

# Filtering/smoothing algorithms
include("algorithms/bootstrap.jl")
include("algorithms/kalman.jl")
include("algorithms/forward.jl")
include("algorithms/rbpf.jl")

end
