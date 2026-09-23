@testitem "Public particle history constructors and checked appends" begin
    using GeneralisedFilters
    const GF = GeneralisedFilters
    cloud(xs, as=collect(eachindex(xs))) = GF.ParticleDistribution(
        [GF.Particle(x, -log(3.0f0), a) for (x, a) in zip(xs, as)], 0.0f0
    )
    initial = cloud([10, 20, 30])
    first_state = cloud([1.0, 2.0, 3.0], [3, 1, 3])
    tree = ParticleTree(initial; capacity=1)
    @test_throws ArgumentError push!(tree, first_state)
    @test [collect(p) for p in get_ancestry(tree)] == [[10], [20], [30]]
    tree = ParticleTree(initial, first_state; capacity=1)
    dense = DenseParticleContainer(initial, first_state)
    paths() = [collect(p) for p in get_ancestry(tree)]
    @test paths() == [[30, 1.0], [10, 2.0], [30, 3.0]]
    @test eltype(dense.weights[1]) === Float32
    for t in 2:15
        next_state = cloud([10.0t+1, 10.0t+2, 10.0t+3], [2, 2, 1])
        before = paths()
        @test_throws ArgumentError push!(tree, cloud([1.0, 2.0, 3.0], [0, 2, 1]))
        @test_throws DimensionMismatch push!(tree, cloud([1.0, 2.0]))
        @test paths() == before
        @test_throws ArgumentError push!(dense, cloud([1.0, 2.0, 3.0], [4, 2, 1]))
        @test length(dense.states) == t-1
        push!(tree, next_state)
        push!(dense, next_state)
        @test paths() == [collect(get_ancestry(dense, i)) for i in 1:3]
    end
    same = ParticleTree(initial)
    push!(same, cloud([40, 50, 60], [2, 3, 1]))
    @test [collect(p) for p in get_ancestry(same)] == [[20, 40], [30, 50], [10, 60]]
    @test_throws ArgumentError ParticleTree(cloud(Int[]))
    @test_throws ArgumentError ParticleTree(initial; capacity=0)
end

@testitem "History owns buffers and shares state objects" begin
    using GeneralisedFilters
    const GF = GeneralisedFilters
    x0 = [[1.0], [2.0]]
    x1 = [[3.0], [4.0]]
    weights = Float32[-1, -1]
    ancestors = [2, 1]
    tree = ParticleTree(x0, x1, ancestors, 2)
    dense = DenseParticleContainer(x0, x1, weights, ancestors)
    x0[1] = [100.0]
    x1[1] = [200.0]
    ancestors[1] = 1
    weights[1] = -100
    @test get_ancestry(tree)[1].x0 == [2.0]
    @test get_ancestry(tree)[1].xs[1] == [3.0]
    @test get_ancestry(dense, 1).xs[1] == [3.0]
    @test dense.weights[1][1] == -1.0f0
    x1[2][1] = 77
    @test get_ancestry(tree)[2].xs[1][1] == 77
    @test get_ancestry(dense, 2).xs[1][1] == 77
    x2 = [[5.0], [6.0]]
    ws2 = Float32[-2, -2]
    as2 = [1, 2]
    push!(dense, x2, ws2, as2)
    x2[1] = [300.0]
    ws2[1] = -300
    as2[1] = 2
    @test dense.states[2][1] == [5.0]
    @test dense.weights[2][1] == -2.0f0
    @test dense.ancestors[2][1] == 1
end
