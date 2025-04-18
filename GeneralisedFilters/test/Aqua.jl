using Aqua
using GeneralisedFilters

@testset "Aqua.jl QA" begin
    Aqua.test_all(GeneralisedFilters)
end
