@testitem "GPU sparse tree pruning, growth and reuse" tags=[:gpu] begin
    using GeneralisedFilters, CUDA, Random
    const GF = GeneralisedFilters
    CUDA.allowscalar(false)
    let
        n = 1025 # Dead branches converge across several thread blocks.
        rng = Xoshiro(912)
        cloud(xs, as) = GF.ParticleDistribution(
            [GF.Particle(x, 0.0f0, a) for (x, a) in zip(xs, as)], 0.0f0
        )
        initial = Int32.(-collect(1:n))
        first = Float32.(10000 .+ (1:n))
        as = collect(1:n)
        cpu = ParticleTree(cloud(initial, as), cloud(first, as); capacity=1)
        gpu = ParticleTree(CuArray(initial), CuArray(first), CuArray(Int32.(as)), 1)
        dense = DenseParticleContainer(cloud(initial, as), cloud(first, as))
        saved = get_ancestry(gpu, n)
        max_capacity = 0
        reused = false
        for t in 2:24
            # Identity forces growth; alternating families and collapse exercise
            # convergent pruning walks and extensive reuse after growth.
            as = if t <= 5
                collect(1:n)
            elseif t % 4 == 0
                fill(n, n)
            else
                rand(rng, 1:17, n)
            end
            xs = Float32.(10000t .+ (1:n))
            before_slots = Set(Array(gpu.leaves))
            insert!(gpu, CuArray(xs), CuArray(isodd(t) ? Int32.(as) : Int64.(as)))
            push!(cpu, cloud(xs, as))
            push!(dense, cloud(xs, as))
            leaves, parents, counts = Array(gpu.leaves),
            Array(gpu.parents),
            Array(gpu.offspring)
            reused |= !isempty(intersect(before_slots, Set(leaves)))
            max_capacity = max(max_capacity, length(gpu.states))
            occupied = findall(>(0), counts)
            expected_counts = zeros(Int64, length(counts))
            expected_counts[leaves] .= 1
            for j in occupied
                parents[j] > 0 && (expected_counts[parents[j]] += 1)
            end
            @test counts == expected_counts
            reachable = Set{Int64}()
            paths_valid = true
            for leaf in leaves
                j = leaf
                for _ in 1:t
                    if !(1 <= j <= length(parents))
                        paths_valid = false
                        break
                    end
                    push!(reachable, j)
                    j = parents[j]
                end
                paths_valid &= -n <= j <= -1
            end
            @test paths_valid
            @test reachable == Set(occupied)
            cpu_paths = get_ancestry(cpu)
            for i in (1, 257, n)
                @test get_ancestry(gpu, i) == cpu_paths[i] == get_ancestry(dense, i)
            end
        end
        @test max_capacity > n
        @test reused
        @test saved.x0 === initial[n]
        @test saved.xs == [first[n]]
        before = get_ancestry(gpu, 1)
        @test_throws ArgumentError insert!(
            gpu, CUDA.zeros(Float64, n), CuArray(Int32.(1:n))
        )
        @test_throws ArgumentError insert!(
            gpu, CUDA.zeros(Float32, n), CUDA.zeros(Int32, n)
        )
        @test get_ancestry(gpu, 1) == before
    end
end

@testitem "GPU sparse nested states and compact trajectories" tags=[:gpu, :batched] begin
    using GeneralisedFilters, BatchedKernels, CUDA, LinearAlgebra
    const GF = GeneralisedFilters
    CUDA.allowscalar(false)
    struct TaggedState{R,I}
        state::R
        tag::I
    end
    n = 5
    function states(t)
        belief = BatchedStruct(
            GaussianState,
            (;
                μ=BatchedCuVector(CUDA.fill(Float32(t), 2, n)),
                Σ=SharedCuMatrix(CUDA.fill(Float32(2t), 2, 2), n),
            ),
        )
        rb = BatchedStruct(
            RBState, (; x=BatchedCuVector(CUDA.fill(Float32(3t), 1, n)), z=belief)
        )
        BatchedStruct(
            TaggedState, (; state=rb, tag=BatchedCuScalar(CuArray(Int64.(2^40 .+ (1:n)))))
        )
    end
    # A completely different initial structure must not be cast to the later one.
    initial = BatchedCuMatrix(CUDA.fill(9.0f0, 3, 3, n))
    s1 = states(1)
    tree = ParticleTree(initial, s1, CuArray(Int32[5, 4, 3, 2, 1]), 1)
    s1.components.state.components.z.components.Σ.data .= -1
    for t in 2:5
        insert!(tree, states(t), CuArray(Int32[2, 2, 2, 4, 4]))
    end
    path = get_ancestry(tree, 1)
    @test Array(path.x0) == fill(9.0f0, 3, 3)
    @test Array(path[1].state.z.Σ) == fill(2.0f0, 2, 2)
    @test [only(Array(s.state.x)) for s in path.xs] == 3.0f0 .* (1:5)
    @test path[5].tag === Int64(2^40 + 1)
    xstorage = parent(path[3].state.x)
    @test sizeof(xstorage.data) == 5sizeof(Float32)
    @test !Base.mightalias(xstorage, tree.states.components.state.components.x.data)
    insert!(tree, states(6), CuArray(Int64[1, 1, 1, 1, 1]))
    @test only(Array(path[5].state.x)) == 15.0f0
    @test all(p -> length(p) == 7, get_ancestry(tree))

    # Public histories keep RB beliefs; CSMC's recorder projects only the outer x.
    particles(s) = BatchedStruct(
        GF.Particle,
        (;
            state=s,
            log_w=BatchedCuScalar(CUDA.zeros(Float32, n)),
            ancestor=BatchedCuScalar(CuArray(Int32.(1:n))),
        ),
    )
    p0 = GF.ParticleDistribution(particles(states(0).components.state), 0.0f0)
    p1 = GF.ParticleDistribution(particles(states(1).components.state), 0.0f0)
    full = ParticleTree(p0, p1)
    @test get_ancestry(full, 1)[1] isa RBState
    projected = GF._init_tree(p0, p1)
    @test projected.history.states isa BatchedCuVector
    @test GF._init_container(p0, p1) isa DenseParticleContainer
end
