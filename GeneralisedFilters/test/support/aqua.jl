@testitem "Aqua package quality" begin
    using Aqua
    using GeneralisedFilters
    using SSMProblems
    Aqua.test_ambiguities([GeneralisedFilters])
    Aqua.test_unbound_args(GeneralisedFilters)
    Aqua.test_undefined_exports(GeneralisedFilters)
    Aqua.test_project_extras(GeneralisedFilters)
    Aqua.test_deps_compat(GeneralisedFilters)
    # AcceleratedKernels is used only by the optional CUDA extension.
    Aqua.test_stale_deps(GeneralisedFilters; ignore=[:AcceleratedKernels])
    # The hierarchical `StateSpaceModel` shorthand extends a constructor SSMProblems owns
    # using only substrate argument types. GeneralisedFilters co-owns that interface.
    Aqua.test_piracies(GeneralisedFilters; treat_as_own=[StateSpaceModel])
    # Aqua resolves a fresh environment from Project.toml alone, discarding local
    # dependencies. That cannot resolve the unreleased SSMProblems in this monorepo.
    # Keep this check for registered installations; CI still precompiles both local
    # packages, and SSMProblems runs its own persistent-task check.
    sibling = joinpath(pkgdir(GeneralisedFilters), "..", "SSMProblems")
    if isdir(sibling) && realpath(pkgdir(SSMProblems)) == realpath(sibling)
        @test_skip Aqua.test_persistent_tasks(GeneralisedFilters)
    else
        Aqua.test_persistent_tasks(GeneralisedFilters)
    end
end
