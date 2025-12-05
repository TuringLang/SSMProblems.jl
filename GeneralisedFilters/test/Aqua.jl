using Aqua
using GeneralisedFilters
using SSMProblems

@testset "Aqua.jl QA" begin
    # Manually specify tests to allow controlling their behaviour
    # Skip test_deps_compat and test_undocumented_names
    Aqua.test_ambiguities([GeneralisedFilters])
    Aqua.test_unbound_args(GeneralisedFilters)
    Aqua.test_undefined_exports(GeneralisedFilters)
    Aqua.test_project_extras(GeneralisedFilters)
    Aqua.test_stale_deps(GeneralisedFilters)
    Aqua.test_piracies(GeneralisedFilters; treat_as_own=[SSMProblems.simulate_from_dist])
    Aqua.test_persistent_tasks(GeneralisedFilters)
end
