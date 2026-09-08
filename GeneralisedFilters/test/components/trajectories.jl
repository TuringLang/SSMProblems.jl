@testitem "ReferenceTrajectory zero-based array contract" begin
    using GeneralisedFilters
    using StaticArrays

    for r in (ReferenceTrajectory(10, [20, 30]),
              ReferenceTrajectory(:initial, [20, 30]),
              ReferenceTrajectory(SVector(0.0), [SVector(1.0), SVector(2.0)]),
              ReferenceTrajectory(10, Int[]))
        expected = [r.x0, r.xs...]
        @test size(r) == (length(expected),)
        @test firstindex(r) == 0
        @test lastindex(r) == length(r.xs)
        @test collect(eachindex(r)) == collect(0:length(r.xs))
        @test axes(axes(r, 1), 1) == axes(r, 1)
        @test collect(LinearIndices(r)) == collect(0:length(r.xs))
        @test first(LinearIndices(r)) == 0
        @test collect(r) == expected
        @test collect([r[i] for i in eachindex(r)]) == expected
        @test collect([r[i] for i in CartesianIndices(r)]) == expected
        @test collect(map(identity, r)) == expected
        @test map(identity, r) isa ReferenceTrajectory
        @test axes(map(identity, r)) == axes(r)

        copied = copy(r)
        @test copied isa ReferenceTrajectory
        @test copied == r
        @test axes(copied) == axes(r)
        @test collect(copied) == expected
        # Exercise Base's generic linear copy path, the original bounds-error trigger.
        destination = Vector{eltype(r)}(undef, length(r))
        @test copyto!(destination, r) == expected
        @test collect(view(r, 0:lastindex(r))) == expected
        @test_throws BoundsError r[-1]
        @test_throws BoundsError r[length(r)]
    end

    x0 = [0.0]
    xs = [[1.0], [2.0]]
    r = ReferenceTrajectory(x0, xs)
    copied = copy(r)
    @test copied.x0 === x0
    @test copied.xs !== xs
    @test copied[1] === xs[1]
    copied.xs[1] = [7.0]
    @test r[1] == [1.0]
    @test copied[1] == [7.0]
end
