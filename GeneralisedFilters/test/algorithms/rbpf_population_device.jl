@testitem "RBPF device population ownership and shared numerical functions" tags=[
    :gpu, :batched
] begin
    using BatchedKernels, CUDA
    const GF = GeneralisedFilters
    CUDA.allowscalar(false)
    # The numerical definitions are ordinary CPU functions, lifted by BK on GPU.
    mean_for(A, x) = A * x
    drift_for(A, B, xp, xn) = A * xp + B * xn
    n = 3
    host_x = reshape(Float32.(1:6), 2, n)
    host_new = host_x .+ 10
    A = Float32[0.3 0; 0 0.4]
    B = Float32[0.2 0; 0 0.7]
    x = BatchedCuVector(CuArray(host_x))
    new_x = BatchedCuVector(CuArray(host_new))
    a, b = SharedCuMatrix(CuArray(A), n), SharedCuMatrix(CuArray(B), n)
    means = mean_for.(a, x)
    drifts = drift_for.(a, b, x, new_x)
    @test Array(means.data) ≈ hcat((mean_for(A, host_x[:, i]) for i in 1:n)...)
    @test Array(drifts.data) ≈
        hcat((drift_for(A, B, host_x[:, i], host_new[:, i]) for i in 1:n)...)
    belief = BatchedStruct(GaussianState, (; μ=means, Σ=a))
    state = BatchedStruct(RBState, (; x, z=belief))
    ancestry = Int64(2)^40 .+ (1:n)
    particles = BatchedStruct(
        GF.Particle,
        (;
            state,
            log_w=BatchedCuScalar(CUDA.zeros(Float32, n)),
            ancestor=BatchedCuScalar(CuArray(ancestry)),
        ),
    )
    fields = GF._rb_population_fields(particles)
    rebuilt = GF._assemble_rb_population(particles, fields)
    @test rebuilt.components.state.components.x === x
    @test rebuilt.components.state.components.z === belief
    @test rebuilt.components.ancestor === particles.components.ancestor
    @test eltype(rebuilt.components.ancestor) === Int64
    @test Array(rebuilt.components.ancestor.data) == ancestry
    @test_throws DimensionMismatch GF._assemble_rb_population(
        particles, merge(fields, (; ancestor=Int64[]))
    )
    changed = GF._assemble_rb_population(
        particles, merge(fields, (; state=RBState(new_x, belief)))
    )
    @test changed.components.state.components.x === new_x
    @test Array(particles.components.state.components.x.data) == host_x
end
