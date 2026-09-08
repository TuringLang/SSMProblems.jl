using Test
using TestItems
using TestItemRunner

# All CPU algorithm, AD, and integration tests are enabled. GPU execution requires
# CUDA hardware and is checked separately by the GPU workflow.
@run_package_tests filter = ti -> !(:gpu in ti.tags)
