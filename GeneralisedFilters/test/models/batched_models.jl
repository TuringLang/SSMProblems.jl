@testitem "Device Gaussian models and RBPF initialisation" tags = [:gpu, :batched] begin
    using CUDA, BatchedKernels, Random, LinearAlgebra
    using GeneralisedFilters:
        ParticleDistribution, inner_dynamics, inner_observation, inner_prior
    CUDA.allowscalar(false)

    # Example-owned batch resolver: conditional model authors declare which
    # parameters vary, while retaining the existing hierarchical model interface.
    function GeneralisedFilters.inner_dynamics(
        d::GeneralisedFilters.GFTest.InnerDynamics,
        ::Integer,
        xp::BatchedCuVector,
        xn::BatchedCuVector,
    )
        length(xp) == length(xn) || throw(DimensionMismatch("outer batch counts"))
        n = length(xp)
        fields = (;
            A=SharedCuMatrix(d.A, n),
            b=BatchedCuVector(d.C * xp.data .+ d.b),
            Q=SharedCuMatrix(d.Q, n),
        )
        D = LinearGaussianDynamics{eltype(fields.A),eltype(fields.b),eltype(fields.Q)}
        return BatchedStruct{D,typeof(fields)}(fields, n)
    end
    transition_mean(d, x) = d.A * x + d.b
    observation_mean(o, x) = o.H * x + o.c

    # One representative Float32 workload and one Float64 smoke case.
    # Avoid a shape grid: these methods delegate bulk array operations to CUDA.
    for (T, dx, dz, dy, n) in ((Float32, 3, 16, 8, 33), (Float64, 2, 3, 2, 5))
        rng = Xoshiro(17)
        _, cpu = GeneralisedFilters.GFTest.create_dummy_linear_gaussian_model(
            rng, dx, dz, dy, T
        )
        op, od = cpu.prior.outer, cpu.dyn.outer
        ip, id, ob = cpu.prior.inner, cpu.dyn.inner, cpu.obs.inner
        gpu = StateSpaceModel(
            GaussianPrior(CuArray(op.μ0), CuArray(op.Σ0)),
            LinearGaussianDynamics(CuArray(od.A), CuArray(od.b), CuArray(od.Q)),
            GaussianPrior(CuArray(ip.μ0), CuArray(ip.Σ0)),
            GeneralisedFilters.GFTest.InnerDynamics(
                CuArray(id.A), CuArray(id.b), CuArray(id.C), CuArray(id.Q)
            ),
            LinearGaussianObservation(CuArray(ob.H), CuArray(ob.c), CuArray(ob.R)),
        )
        xp = randn(rng, T, dx, n)
        xn = randn(rng, T, dx, n)
        z = randn(rng, T, dz, n)
        xpg = BatchedCuVector(CuArray(xp))
        xng = BatchedCuVector(CuArray(xn))
        zg = BatchedCuVector(CuArray(z))
        d = inner_dynamics(gpu, 7, xpg, xng)
        o = inner_observation(gpu, 7, xng)
        p = inner_prior(gpu, xpg)
        @test d.components.A.data === gpu.dyn.inner.A
        @test d.components.Q.data === gpu.dyn.inner.Q
        @test p.components.μ0.data === gpu.prior.inner.μ0
        @test isconcretetype(eltype(d))
        @test Array(fuse(transition_mean, d, zg).data) ≈ id.A * z .+ id.C * xp .+ id.b
        @test Array(fuse(observation_mean, o, zg).data) ≈ ob.H * z .+ ob.c
        constant = inner_dynamics(
            gpu.dyn.outer, 7, xpg, SharedCuVector(CuArray(xn[:, 1]), n)
        )
        @test constant.components.A.data === gpu.dyn.outer.A
        @test constant.components.Q isa SharedCuMatrix

        eps = Array(randn(CUDA.RNG(91), T, (dx, n)))
        sample = simulate(CUDA.RNG(91), gpu.dyn.outer, 7, xpg)
        @test Array(sample.data) ≈ od.A * xp .+ od.b .+ cholesky(Symmetric(od.Q)).L * eps
        @test Array(xpg.data) == xp
        @test Array(gpu.dyn.outer.Q) == od.Q

        algo = RBPF(BF(n), KF())
        state = initialise(CUDA.RNG(91), gpu.prior, algo)
        leaves = state.particles.components
        @test state isa ParticleDistribution{T}
        @test isconcretetype(eltype(state.particles))
        @test state.ll_baseline === zero(T)
        @test Array(GeneralisedFilters.log_weights(state)) == zeros(T, n)
        @test Array(leaves.ancestor.data) == zeros(Int32, n)
        @test Array(leaves.state.components.x.data) ≈
            op.μ0 .+ cholesky(Symmetric(op.Σ0)).L * eps
        @test Array(leaves.state.components.z.components.μ.data) ≈ repeat(ip.μ0, 1, n)
        @test Array(leaves.state.components.z.components.Σ.data) ≈ repeat(ip.Σ0, 1, 1, n)
        @test Array(gpu.prior.outer.Σ0) == op.Σ0
        if T === Float32
            # Storage dispatch must not silently accept a host RNG or unsupported
            # host reference states; these checks are independent of precision.
            @test_throws ArgumentError simulate(rng, gpu.dyn.outer, 7, xpg)
            @test_throws DimensionMismatch inner_dynamics(
                gpu.dyn.outer, 7, xpg, BatchedCuVector(CUDA.zeros(T, dx, n + 1))
            )
            @test_throws ArgumentError initialise(rng, gpu.prior, algo)
            @test_throws ArgumentError initialise(
                CUDA.RNG(91), gpu.prior, algo; ref_state=[zeros(T, dx)]
            )
            # Shared model parameters must become independently owned beliefs.
            leaves.state.components.z.components.μ.data .= zero(T)
            leaves.state.components.z.components.Σ.data .= zero(T)
            @test Array(gpu.prior.inner.μ0) == ip.μ0
            @test Array(gpu.prior.inner.Σ0) == ip.Σ0
            @test initialise(rng, cpu.prior, algo).particles isa Vector
        end
    end

    # Conditional priors can return distinct model-owned batched beliefs.
    struct ConditionalDevicePrior{M,C}
        μ::M
        Σ::C
    end
    function GeneralisedFilters.inner_prior(p::ConditionalDevicePrior, x::BatchedCuVector)
        fs = (; μ0=BatchedCuVector(p.μ), Σ0=BatchedCuMatrix(p.Σ))
        P = GaussianPrior{eltype(fs.μ0),eltype(fs.Σ0)}
        return BatchedStruct{P,typeof(fs)}(fs, length(x))
    end
    means = reshape(Float32.(1:10), 2, 5)
    covs = cat((Float32(i) * Matrix{Float32}(I, 2, 2) for i in 1:5)...; dims=3)
    cp = ConditionalDevicePrior(CuArray(means), CuArray(covs))
    # Zero noise and a rank-deficient rectangular root cover non-SPD covariances.
    for rank in (0, 1)
        factor = reshape(Float32.(1:(2 * rank)), 2, rank) ./ 10.0f0
        op = GaussianPrior(CUDA.ones(Float32, 2), CovarianceFactor(CuArray(factor)))
        hp = HierarchicalPrior(op, cp)
        state = initialise(CUDA.RNG(123), hp, RBPF(BF(5), KF()))
        leaves = state.particles.components.state.components
        noise = Array(randn(CUDA.RNG(123), Float32, (rank, 5)))
        @test Array(leaves.x.data) ≈ ones(Float32, 2) .+ factor * noise
        if rank == 1
            # A conditional prior must preserve distinct particle beliefs and
            # copy model-owned batch storage before the filter mutates it.
            @test Array(leaves.z.components.μ.data) == means
            @test Array(leaves.z.components.Σ.data) == covs
            leaves.z.components.μ.data .= 0.0f0
            leaves.z.components.Σ.data .= 0.0f0
            @test Array(cp.μ) == means
            @test Array(cp.Σ) == covs
        end
        dyn = LinearGaussianDynamics(
            CuArray(Matrix{Float32}(I, 2, 2)), CUDA.zeros(Float32, 2), op.Σ0
        )
        draw = simulate(CUDA.RNG(123), dyn, 1, leaves.x)
        @test Array(draw.data) ≈ Array(leaves.x.data) .+ factor * noise
    end
    badcp = ConditionalDevicePrior(CUDA.zeros(Float32, 2, 4), CuArray(covs))
    @test_throws DimensionMismatch initialise(
        CUDA.RNG(1),
        HierarchicalPrior(
            GaussianPrior(CUDA.zeros(Float32, 2), CuArray(Matrix{Float32}(I, 2, 2))), badcp
        ),
        RBPF(BF(5), KF()),
    )

    x = BatchedCuVector(CUDA.zeros(Float32, 2, 3))
    wrong_type = GaussianPrior(CUDA.zeros(Float64, 2), CuArray(Matrix{Float64}(I, 2, 2)))
    @test_throws ArgumentError inner_prior(wrong_type, x)
    wrong_shape = GaussianPrior(CUDA.zeros(Float32, 2), CUDA.zeros(Float32, 3, 3))
    @test_throws DimensionMismatch inner_prior(wrong_shape, x)

    # Check the fused affine draw against independent arithmetic with the same
    # Philox draws; replay alone would not catch a wrong covariance orientation.
    factor = Float32[0.2 0; 0.1 0.3]
    μ0 = Float32[0.1, -0.2]
    prior = HierarchicalPrior(
        GaussianPrior(CuArray(μ0), CovarianceFactor(CuArray(factor))),
        GaussianPrior(CUDA.zeros(Float32, 1), CUDA.ones(Float32, 1, 1)),
    )
    rng = BatchedRNG(29)
    sample_noise(r) = randn(r, Float32, 2)
    noise = Array(fuse(sample_noise, copy(rng); batch_size=5).data)
    initial = initialise(rng, prior, RBPF(BF(5), KF()))
    x = initial.particles.components.state.components.x
    @test Array(x.data) ≈ μ0 .+ factor * noise
    checkpoint = copy(rng)
    noise = Array(fuse(sample_noise, copy(rng); batch_size=5).data)
    A = Float32[0.8 0.1; 0 0.9]
    dyn = LinearGaussianDynamics(CuArray(A), CuArray(μ0), CovarianceFactor(CuArray(factor)))
    draw = simulate(rng, dyn, 1, x)
    @test Array(draw.data) ≈ A * Array(x.data) .+ μ0 .+ factor * noise
    @test Array(simulate(checkpoint, dyn, 1, x).data) == Array(draw.data)
end
