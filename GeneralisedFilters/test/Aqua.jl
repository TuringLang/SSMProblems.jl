module AquaTests

using Aqua: Aqua
using GeneralisedFilters

@testset "Aqua.jl QA" begin
    Aqua.test_all(GeneralisedFilters)
end

end
