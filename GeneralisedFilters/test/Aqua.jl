module AquaTests

using Aqua: Aqua
using GeneralisedFilters

@testset "Aqua.jl QA" begin
    Aqua.test_all(SSMProblems)
end

end
