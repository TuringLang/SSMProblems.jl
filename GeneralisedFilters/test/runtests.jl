using Test
using TestItems
using TestItemRunner

# `@run_package_tests` discovers every `@testitem` in the package, so test suites are
# selected here by filtering rather than by `include`.
#
# Extension/hardware filtering:
# - :gpu tests require CUDA hardware
# - :mooncake tests require Mooncake (loaded as test dep, triggers MooncakeExt)
#
# The suites below cover algorithms and integrations whose ports land in later stages of the
# interface redesign; they are excluded by filename until their components exist.
const PENDING_TEST_FILES = (
    "algorithms/particles.jl",
    "algorithms/rbpf.jl",
    "algorithms/csmc.jl",
    "integrations/logdensity.jl",
    "integrations/particle_gibbs.jl",
    "integrations/turing.jl",
)

_is_pending(filename) = any(f -> endswith(filename, f), PENDING_TEST_FILES)

@run_package_tests filter = ti -> !(:gpu in ti.tags) && !_is_pending(ti.filename)
