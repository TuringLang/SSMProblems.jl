module AquaTests

using Aqua: Aqua
using GeneralisedFilters

@testitem "Aqua.jl QA" begin
    Aqua.test_all(GeneralisedFilters)
end

end
