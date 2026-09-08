@testitem "Aqua package quality" begin
    using Aqua
    using GeneralisedFilters
    Aqua.test_ambiguities([GeneralisedFilters])
    Aqua.test_unbound_args(GeneralisedFilters)
    Aqua.test_undefined_exports(GeneralisedFilters)
    Aqua.test_project_extras(GeneralisedFilters)
    Aqua.test_deps_compat(GeneralisedFilters)
    # AcceleratedKernels is used only by the optional CUDA extension.
    Aqua.test_stale_deps(GeneralisedFilters; ignore=[:AcceleratedKernels])
    Aqua.test_piracies(GeneralisedFilters)
    Aqua.test_persistent_tasks(GeneralisedFilters)
end
