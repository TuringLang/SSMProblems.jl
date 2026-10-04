using TestItems

@testitem "GPU composite particle storage" tags = [:gpu, :batched] begin
    using Test
    using CUDA
    using BatchedKernels
    using GeneralisedFilters
    using GeneralisedFilters: Particle, ParticleDistribution, RBState
    using Random: Xoshiro

    CUDA.allowscalar(false)

    # One non-warp-aligned population exercises the nested storage path.
    # Leaf indexing is delegated to CUDA, so warp-boundary size grids add no
    # coverage of filtering-specific code here.
    let n = 33
        x_host = reshape(Float32.(1:(2n)), 2, n)
        μ_host = reshape(Float32.(1:(3n)) ./ 10, 3, n)
        Σ_host = reshape(Float32.(1:(9n)) ./ 100, 3, 3, n)
        x = BatchedCuVector(CuArray(x_host))
        μ = BatchedCuVector(CuArray(μ_host))
        Σ = BatchedCuMatrix(CuArray(Σ_host))
        gaussian_fields = (; μ, Σ)
        G = GaussianState{eltype(μ),eltype(Σ)}
        z = BatchedStruct{G,typeof(gaussian_fields)}(gaussian_fields, n)
        state_fields = (; x, z)
        S = RBState{eltype(x),eltype(z)}
        states = BatchedStruct{S,typeof(state_fields)}(state_fields, n)
        w_host = -Float32.(0:(n - 1)) ./ 10
        w = BatchedCuScalar(CuArray(w_host))
        # Ancestor metadata must remain exact beyond Float32's integer range.
        a_host = Int64(2)^24 .+ collect(Int64, 1:n)
        a = BatchedCuScalar(CuArray(a_host))
        fields = (; state=states, log_w=w, ancestor=a)
        P = Particle{eltype(states),Float32,Int64}
        particles = BatchedStruct{P,typeof(fields)}(fields, n)
        distribution = ParticleDistribution(particles, 0.0f0)
        @test GeneralisedFilters.log_weights(distribution) === w.data
        probabilities = GeneralisedFilters.get_weights(distribution)
        @test probabilities isa CuVector{Float32}
        @test Array(probabilities) ≈ exp.(w_host .- GeneralisedFilters.logsumexp(w_host))

        preserved = GeneralisedFilters.preserve_sample(distribution)
        @test preserved.particles.components.state === states
        @test GeneralisedFilters.log_weights(preserved) === w.data
        @test Array(preserved.particles.components.ancestor.data) == collect(1:n)
        @test Array(a.data) == a_host
        @test preserved.ll_baseline === distribution.ll_baseline

        # One map combines duplicates, reordering and both boundary indices.
        let indices = [n; collect(1:(n - 2)); n]
            gpu_indices = CuArray(Int32.(indices))
            gathered = GeneralisedFilters.construct_new_state(
                distribution, gpu_indices, nothing
            )
            c = gathered.particles.components
            @test eltype(gathered.particles) === P
            @test Array(c.state.components.x.data) == x_host[:, indices]
            @test Array(c.state.components.z.components.μ.data) == μ_host[:, indices]
            @test Array(c.state.components.z.components.Σ.data) == Σ_host[:, :, indices]
            @test Array(c.log_w.data) == zeros(Float32, n)
            @test Array(c.ancestor.data) == indices
            @test c.ancestor.data !== gpu_indices
            @test gathered.ll_baseline === 0.0f0
            # A subsequent writer must not corrupt the previous population.
            c.state.components.z.components.Σ.data .= -1
            c.state.components.x.data .= -1
            @test Array(Σ.data) == Σ_host
            @test Array(x.data) == x_host
        end

        shifted_fields = merge(fields, (; log_w=BatchedCuScalar(CUDA.fill(-1.0f8, n))))
        shifted_particles = BatchedStruct{P,typeof(shifted_fields)}(shifted_fields, n)
        shifted_distribution = ParticleDistribution(shifted_particles, 0.0f0)
        @test Array(GeneralisedFilters.get_weights(shifted_distribution)) ≈
            fill(inv(Float32(n)), n)

        shifted_normalised, shifted_ll = GeneralisedFilters.marginalise!(
            shifted_distribution, shifted_particles
        )
        @test eltype(GeneralisedFilters.log_weights(shifted_normalised)) === Float32
        @test Array(GeneralisedFilters.log_weights(shifted_normalised)) ≈
            fill(-log(Float32(n)), n)
        @test sum(exp, GeneralisedFilters.log_weights(shifted_normalised)) ≈ 1.0f0
        @test shifted_ll === GeneralisedFilters.logsumexp(fill(-1.0f8, n))

        # Exercise existing ESS orchestration, not just storage helpers.
        skipped = GeneralisedFilters.maybe_resample(
            Xoshiro(1), ESSResampler(0.0), distribution
        )
        @test skipped.particles.components.state === states
        @test Array(skipped.particles.components.ancestor.data) == collect(1:n)
        resampled = GeneralisedFilters.maybe_resample(
            Xoshiro(1), ESSResampler(1.0), distribution
        )
        resampled_ancestors = Array(resampled.particles.components.ancestor.data)
        @test all(i -> 1 <= i <= n, resampled_ancestors)
        @test Array(resampled.particles.components.state.components.x.data) ==
            x_host[:, resampled_ancestors]
        @test Array(GeneralisedFilters.log_weights(resampled)) == zeros(Float32, n)

        baseline = GeneralisedFilters.logsumexp(w_host)
        incoming = ParticleDistribution(particles, baseline)
        likelihoods = Float32.(1:n) ./ 20
        next_fields = merge(
            fields, (; log_w=BatchedCuScalar(w.data .+ CuArray(likelihoods)))
        )
        weighted = BatchedStruct{P,typeof(next_fields)}(next_fields, n)
        normalised, increment = GeneralisedFilters.marginalise!(incoming, weighted)
        lse = GeneralisedFilters.logsumexp(w_host .+ likelihoods)
        @test increment ≈ lse - baseline atol = 2.0f-6
        @test Array(GeneralisedFilters.log_weights(normalised)) ≈
            w_host .+ likelihoods .- lse
        @test normalised.ll_baseline === 0.0f0
        @test normalised.particles.components.state === states
        @test Array(a.data) == a_host
        @test Array(w.data) == w_host
        @test Array(weighted.components.log_w.data) ≈ w_host .+ likelihoods
        @test_throws BoundsError GeneralisedFilters.construct_new_state(
            distribution, CuArray(fill(0, n)), nothing
        )
        @test_throws DimensionMismatch GeneralisedFilters.construct_new_state(
            distribution, CUDA.zeros(Int, n + 1), nothing
        )
        @test_throws ArgumentError GeneralisedFilters.construct_new_state(
            distribution, CuArray(1:n), CUDA.zeros(Float32, n)
        )

        @test_throws ArgumentError GeneralisedFilters.construct_new_state(
            ParticleDistribution(particles, GeneralisedFilters.TypelessBaseline(n)),
            CuArray(1:n),
            nothing,
        )
        # Explicitly shared covariance stays shared during ancestor gathering.
        shared_covariance = SharedCuMatrix(CuArray(Σ_host[:, :, 1]), n)
        shared_gaussian_fields = (; μ, Σ=shared_covariance)
        shared_z = BatchedStruct{G,typeof(shared_gaussian_fields)}(
            shared_gaussian_fields, n
        )
        shared_state_fields = (; x, z=shared_z)
        shared_states = BatchedStruct{S,typeof(shared_state_fields)}(shared_state_fields, n)
        shared_fields = merge(fields, (; state=shared_states))
        shared_particles = BatchedStruct{P,typeof(shared_fields)}(shared_fields, n)
        shared_distribution = ParticleDistribution(shared_particles, 0.0f0)
        shared_gathered = GeneralisedFilters.construct_new_state(
            shared_distribution, CuArray(fill(n, n)), nothing
        )
        @test shared_gathered.particles.components.state.components.z.components.Σ.data ===
            shared_covariance.data
        if n > 1
            partial_weights = fill(-Inf32, n)
            partial_weights[1] = 0.0f0
            partial_fields = merge(
                fields, (; log_w=BatchedCuScalar(CuArray(partial_weights)))
            )
            partial_particles = BatchedStruct{P,typeof(partial_fields)}(partial_fields, n)
            partial_distribution = ParticleDistribution(partial_particles, 0.0f0)
            @test Array(GeneralisedFilters.get_weights(partial_distribution)) ==
                [1.0f0; zeros(Float32, n - 1)]
            partial_normalised, partial_ll = GeneralisedFilters.marginalise!(
                partial_distribution, partial_particles
            )
            @test partial_ll === 0.0f0
            @test Array(GeneralisedFilters.log_weights(partial_normalised)) ==
                partial_weights
        end

        for invalid in (NaN32, Inf32, -Inf32)
            bad_fields = merge(fields, (; log_w=BatchedCuScalar(CUDA.fill(invalid, n))))
            bad = BatchedStruct{P,typeof(bad_fields)}(bad_fields, n)
            bad_distribution = ParticleDistribution(bad, 0.0f0)
            @test_throws ArgumentError GeneralisedFilters.get_weights(bad_distribution)
            @test_throws ArgumentError GeneralisedFilters.marginalise!(incoming, bad)
        end
    end
end
