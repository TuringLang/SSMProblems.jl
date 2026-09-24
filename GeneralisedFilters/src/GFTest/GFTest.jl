module GFTest

using Distributions
using LinearAlgebra
using Random
using StaticArrays

using GeneralisedFilters

include("utils.jl")
include("gradients.jl")
include("models/linear_gaussian.jl")
include("models/mixture_observation.jl")
include("models/dummy_linear_gaussian.jl")
include("models/dummy_discrete.jl")
include("proposals.jl")
include("resamplers.jl")

export MixtureObservation
export check_gradients, central_diff

end
