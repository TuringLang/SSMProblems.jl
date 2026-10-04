@testitem "Batched RBPF steps share the CPU Kalman equations" tags = [:gpu, :batched] begin
    using CUDA, BatchedKernels, Random, LinearAlgebra
    using GeneralisedFilters:
        GaussianState, kalman_predict, kalman_update, step, predict, update
    CUDA.allowscalar(false)

    # The application declares the conditional batching rule; constant A and Q
    # stay shared while the drift depends on the previous outer particle.
    struct ConditionalBatchDynamics{D}
        dynamics::D
    end
    function GeneralisedFilters.inner_dynamics(
        wrapper::ConditionalBatchDynamics,
        ::Integer,
        xp::BatchedCuVector,
        xn::BatchedCuVector,
    )
        d = wrapper.dynamics
        n = length(xp)
        fs = (;
            A=SharedCuMatrix(d.A, n),
            b=BatchedCuVector(d.C * xp.data .+ d.b),
            Q=SharedCuMatrix(d.Q, n),
        )
        D = LinearGaussianDynamics{eltype(fs.A),eltype(fs.b),eltype(fs.Q)}
        return BatchedStruct{D,typeof(fs)}(fs, n)
    end

    let
        T, dx, dz, dy, n = Float32, 2, 16, 5, 19
        rng = Xoshiro(72)
        _, cpu = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
            rng, dx, dz, dy, T
        )
        op, od, ip, id, ob = cpu.prior.outer,
        cpu.dyn.outer, cpu.prior.inner, cpu.dyn.inner,
        cpu.obs.inner
        # The dummy generator uses positive unscaled A/H matrices; at D=16 its
        # innovation covariance can have condition number >1e5 in Float32. Use a
        # stable, well-conditioned instance to compare the shared equations tightly.
        od.A ./= T(1.1) * opnorm(od.A)
        id.A ./= T(1.1) * opnorm(id.A)
        ob.H ./= sqrt(T(dz))
        ip.Σ0 .+= T(0.1) .* Matrix{T}(I, dz, dz)
        id.Q .+= T(0.01) .* Matrix{T}(I, dz, dz)
        ob.R .+= T(0.1) .* Matrix{T}(I, dy, dy)
        gpu = StateSpaceModel(
            GaussianPrior(CuArray(op.μ0), CuArray(op.Σ0)),
            LinearGaussianDynamics(CuArray(od.A), CuArray(od.b), CuArray(od.Q)),
            GaussianPrior(CuArray(ip.μ0), CuArray(ip.Σ0)),
            ConditionalBatchDynamics(
                GeneralisedFilters.GFTest.InnerDynamics(
                    CuArray(id.A), CuArray(id.b), CuArray(id.C), CuArray(id.Q)
                ),
            ),
            LinearGaussianObservation(CuArray(ob.H), CuArray(ob.c), CuArray(ob.R)),
        )
        state = initialise(CUDA.RNG(21), gpu.prior, RBPF(BF(n), KF()))
        previous_x = Array(state.particles.components.state.components.x.data)
        beliefs = [GaussianState(copy(ip.μ0), copy(ip.Σ0)) for _ in 1:n]
        logweights = zeros(T, n)
        total_cpu = zero(T)
        total_gpu = zero(T)

        # Skip, resample, then skip after nonuniform weighting. Clone only the GPU
        # noise stream; all conditional Kalman updates and evidence are computed on CPU.
        for (t, threshold) in enumerate((0.0, 1.0, 0.0))
            algo = RBPF(BF(n; threshold), KF())
            ref_rng = CUDA.RNG(100 + t)
            indices = if threshold == 1.0
                Array(
                    GeneralisedFilters.sample_ancestors(
                        ref_rng, Systematic(), GeneralisedFilters.get_weights(state)
                    ),
                )
            else
                collect(1:n)
            end
            xp = previous_x[:, indices]
            noise = Array(randn(ref_rng, T, (dx, n)))
            expected_x = od.A * xp .+ od.b .+ cholesky(Symmetric(od.Q)).L * noise
            y = randn(rng, T, dy)
            incoming_weights = threshold == 1.0 ? zeros(T, n) : logweights
            baseline = GeneralisedFilters._weight_logsumexp(incoming_weights)
            increments = zeros(T, n)
            next_beliefs = map(1:n) do i
                d = GeneralisedFilters.inner_dynamics(cpu, t, xp[:, i], expected_x[:, i])
                z, ll = kalman_update(kalman_predict(beliefs[indices[i]], d), ob, y)
                increments[i] = ll
                return z
            end
            rawweights = incoming_weights .+ increments
            normalizer = GeneralisedFilters._weight_logsumexp(rawweights)
            logweights = rawweights .- normalizer
            total_cpu += normalizer - baseline

            state, ll = step(CUDA.RNG(100 + t), gpu, algo, t, state, CuArray(y))
            fs = state.particles.components
            @test Array(fs.ancestor.data) == indices
            @test Array(fs.state.components.x.data) ≈ expected_x
            @test Array(fs.state.components.z.components.μ.data) ≈
                hcat((z.μ for z in next_beliefs)...)
            @test Array(fs.state.components.z.components.Σ.data) ≈
                cat((z.Σ for z in next_beliefs)...; dims=3)
            @test Array(GeneralisedFilters.log_weights(state)) ≈ logweights rtol=2.0f-5 atol=2.0f-5
            total_gpu += ll
            beliefs, previous_x = next_beliefs, expected_x
        end
        @test total_gpu ≈ total_cpu rtol=2.0f-5

        # Reject an incomplete reference before conditional resampling starts.
        algo = RBPF(BF(n), KF())
        @test_throws ArgumentError step(
            CUDA.RNG(1), gpu, algo, 4, state, CUDA.zeros(T, dy); ref_state=[zeros(T, dx)]
        )
        @test_throws ArgumentError update(gpu.obs, algo, 4, state, zeros(T, dy))
    end
end
