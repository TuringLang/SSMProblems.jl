module GaelResearch

include("utils.jl")
include("models.jl")
include("algorithms.jl")

using .Utils
using .Models
using .Algorithms

export rand_cov, ensure_posdef, estimate_particle_count
export ParameterisedSSM
export PMMH, PGibbs, EHMM

end
