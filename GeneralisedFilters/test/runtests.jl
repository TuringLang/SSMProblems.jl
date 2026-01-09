using Test
using TestItems
using TestItemRunner

@run_package_tests filter = ti -> !(:gpu in ti.tags)

# Algorithm tests (by family)
include("algorithms/kalman.jl")
include("algorithms/particles.jl")
include("algorithms/discrete.jl")
include("algorithms/rbpf.jl")
include("algorithms/csmc.jl")

# Component tests
include("components/resamplers.jl")
include("components/kalman_gradient.jl")

# Quality tests
include("support/type_stability.jl")
include("support/aqua.jl")
