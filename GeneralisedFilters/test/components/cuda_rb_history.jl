@testitem "Sparse full RB history agrees with dense genealogy" tags=[:gpu, :batched] begin
    using GeneralisedFilters, BatchedKernels, CUDA, LinearAlgebra, Random
    const GF = GeneralisedFilters
    CUDA.allowscalar(false)

    function populations(t, ancestors, d)
        n = length(ancestors)
        # Every field distinguishes particle identity and time. Covariances are
        # positive definite and include off-diagonal entries, not just diagonals.
        x = [Float32(t + i / 4096) for k in 1:2, i in 1:n]
        μ = [Float32(t + i / 2048 + k / 32) for k in 1:d, i in 1:n]
        Σ = Array{Float32}(undef, d, d, n)
        for i in 1:n, c in 1:d, r in 1:d
            Σ[r, c, i] =
                r == c ? Float32(2 + t + i / 2048 + r / 16) : Float32((r + c) / 1024)
        end
        cpu = GF.ParticleDistribution(
            [
                GF.Particle(
                    RBState(x[:, i], GaussianState(μ[:, i], Σ[:, :, i])),
                    0.0f0,
                    ancestors[i],
                ) for i in 1:n
            ],
            0.0f0,
        )
        belief = BatchedStruct(
            GaussianState, (; μ=BatchedCuVector(CuArray(μ)), Σ=BatchedCuMatrix(CuArray(Σ)))
        )
        state = BatchedStruct(RBState, (; x=BatchedCuVector(CuArray(x)), z=belief))
        gpu = GF.ParticleDistribution(
            BatchedStruct(
                GF.Particle,
                (;
                    state,
                    log_w=BatchedCuScalar(CUDA.zeros(Float32, n)),
                    ancestor=BatchedCuScalar(CuArray(ancestors)),
                ),
            ),
            0.0f0,
        )
        return cpu, gpu
    end

    host(s) = RBState(Array(s.x), GaussianState(Array(s.z.μ), Array(s.z.Σ)))
    same(a, b) = a.x == b.x && a.z.μ == b.z.μ && a.z.Σ == b.z.Σ

    let
        n, d = 1025, 16
        rng = Xoshiro(720)
        ci, gi = populations(0, Int32.(1:n), d)
        c1, g1 = populations(1, Int32.(1:n), d)
        dense = DenseParticleContainer(ci, c1)
        sparse = ParticleTree(gi, g1; capacity=1)
        saved = get_ancestry(sparse, n)
        saved_host = map(host, saved)
        reused = false
        for t in 2:12
            ancestors = if t <= 4
                Int32.(1:n)
            elseif t % 3 == 0
                fill(Int32(n), n)
            else
                rand(rng, Int32(1):Int32(37), n)
            end
            cpu, gpu = populations(t, ancestors, d)
            old_slots = Set(Array(sparse.leaves))
            push!(dense, cpu)
            push!(sparse, gpu)
            reused |= !isempty(intersect(old_slots, Set(Array(sparse.leaves))))
            # Mutating a source population after insertion must not alter any
            # retained outer state, mean or covariance.
            fields = gpu.particles.components.state.components
            fields.x.data .= -100
            fields.z.components.μ.data .= -200
            fields.z.components.Σ.data .= -300
            for i in (1, 257, n)
                actual = map(host, get_ancestry(sparse, i))
                expected = get_ancestry(dense, i)
                @test all(same(actual[k], expected[k]) for k in 0:t)
            end
        end
        @test reused
        @test length(sparse.states) > n
        @test all(same(host(saved[k]), saved_host[k]) for k in 0:1)
        @test sizeof(parent(saved[1].z.Σ).data) == d * d * sizeof(Float32)
    end
end
