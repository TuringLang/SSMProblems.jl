module GFTest

using Distributions
using LinearAlgebra
using PDMats
using Random
using StaticArrays

using GeneralisedFilters

include("utils.jl")
include("models/linear_gaussian.jl")
include("models/mixture_observation.jl")

export MixtureObservation

end
