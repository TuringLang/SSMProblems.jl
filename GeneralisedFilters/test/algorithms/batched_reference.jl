@testitem "Batched RBPF reference conditioning" tags = [:gpu, :batched] begin
    using CUDA, BatchedKernels, Random, LinearAlgebra
    const GF = GeneralisedFilters
    CUDA.allowscalar(false)

    # Initial belief depends on x0; transition depends on BOTH gathered xp and pinned xn.
    struct ReferenceInner{M}
        initial::M
        previous::M
        current::M
    end
    ReferenceInner() =
        ReferenceInner(fill(0.3f0, 1, 1), fill(0.2f0, 1, 1), fill(0.7f0, 1, 1))
    # One numerical definition per operation, called on CPU values or traced by BK.
    reference_mean(A, x) = A * x
    reference_drift(A, B, xp, xn) = A * xp + B * xn
    GF.inner_prior(d::ReferenceInner, x::AbstractVector) =
        GaussianPrior(reference_mean(d.initial, x), ones(Float32, 1, 1))
    GF.inner_dynamics(
        d::ReferenceInner, ::Integer, xp::AbstractVector, xn::AbstractVector
    ) = LinearGaussianDynamics(
        fill(0.5f0, 1, 1),
        reference_drift(d.previous, d.current, xp, xn),
        fill(0.1f0, 1, 1),
    )
    function GF.inner_prior(d::ReferenceInner, x::BatchedCuVector)
        n = length(x)
        return BatchedStruct(
            GaussianPrior,
            (;
                μ0=reference_mean.(SharedCuMatrix(d.initial, n), x),
                Σ0=SharedCuMatrix(CUDA.ones(Float32, 1, 1), n),
            ),
        )
    end
    function GF.inner_dynamics(
        d::ReferenceInner,
        ::Integer,
        xp::BatchedCuVector,
        xn::Union{BatchedCuVector,SharedCuVector},
    )
        n = length(xp)
        return BatchedStruct(
            LinearGaussianDynamics,
            (;
                A=SharedCuMatrix(CUDA.fill(0.5f0, 1, 1), n),
                b=reference_drift.(
                    SharedCuMatrix(d.previous, n), SharedCuMatrix(d.current, n), xp, xn
                ),
                Q=SharedCuMatrix(CUDA.fill(0.1f0, 1, 1), n),
            ),
        )
    end
    let
        inner = ReferenceInner(
            CUDA.fill(0.3f0, 1, 1), CUDA.fill(0.2f0, 1, 1), CUDA.fill(0.7f0, 1, 1)
        )
        model = StateSpaceModel(
            GaussianPrior(CUDA.zeros(Float32, 1), CUDA.ones(Float32, 1, 1)),
            LinearGaussianDynamics(
                CUDA.fill(0.8f0, 1, 1),
                CUDA.zeros(Float32, 1),
                CovarianceFactor(CUDA.fill(0.2f0, 1, 1)),
            ),
            inner,
            inner,
            LinearGaussianObservation(
                CUDA.ones(Float32, 1, 1), CUDA.zeros(Float32, 1), CUDA.fill(0.2f0, 1, 1)
            ),
        )
        n = 7
        refs = Float32[0.4, -0.7, 0.9, -0.2, 0.6]
        storage = CuArray(reshape(refs, 1, :))
        vectors = [view(storage, :, i) for i in eachindex(refs)]
        reference = ReferenceTrajectory(vectors[1], vectors[2:end])
        algo = RBPF(BF(n), KF())
        state = initialise(CUDA.RNG(9), model.prior, algo; ref_state=reference)
        initial = state
        # A shared next state must not make the candidate-parent contribution shared.
        parents = initial.particles.components.state.components.x
        dynamics = GF.inner_dynamics(model.dyn, 1, parents, reference[1])
        @test Array(dynamics.components.b.data) ≈
            0.2f0 .* Array(parents.data) .+ 0.7f0 * refs[2]
        history = nothing
        sparse = nothing
        population_x = [Array(state.particles.components.state.components.x.data)]
        population_ancestors = Vector{Int}[]
        xp = Array(state.particles.components.state.components.x.data)
        @test xp[1, 1] == refs[1]
        @test Array(state.particles.components.state.components.z.components.μ.data) ≈
            0.3f0 .* xp
        initial_bk = initialise(BatchedRNG(9), model.prior, algo; ref_state=vectors)
        @test Array(initial_bk.particles.components.state.components.x.data)[1, 1] ==
            refs[1]
        beliefs = [GaussianState([0.3f0 * xp[1, i]], ones(Float32, 1, 1)) for i in 1:n]
        weights = zeros(Float32, n)
        obs = LinearGaussianObservation(
            ones(Float32, 1, 1), zeros(Float32, 1), fill(0.2f0, 1, 1)
        )
        for (t, threshold) in enumerate((0.0, 1.0, 1.0, 0.0))
            algo = RBPF(BF(n; threshold), KF())
            y = Float32[0.15t]
            old_x = copy(xp)
            old_state = state
            if t == 3
                # Future ancestor sampling must gather the chosen Gaussian belief,
                # while the NEW reference outer state stays in output slot one.
                state = GF.resample(
                    CUDA.RNG(30), Systematic(), state; ref_state=reference, ref_idx=n
                )
                @test Array(state.particles.components.ancestor.data)[1] == n
                state, ll = GF.move(
                    BatchedRNG(31), model, algo, t, state, CuArray(y); ref_state=reference
                )
            else
                state, ll = GF.step(
                    CUDA.RNG(20 + t), model, algo, t, state, CuArray(y); ref_state=reference
                )
            end
            if t == 1
                history = DenseParticleContainer(initial, state)
                sparse = GF._init_tree(initial, state)
            else
                push!(history, state)
                GF._update_tree!(sparse, state)
            end
            projected = GF._rb_population_fields(state.particles)
            reassembled = GF._assemble_rb_population(state.particles, projected)
            @test reassembled.components.state.components.x === projected.state.x
            @test reassembled.components.state.components.z === projected.state.z
            @test reassembled.components.ancestor === projected.ancestor
            @test eltype(projected.ancestor) === Int32
            @test_throws DimensionMismatch GF._assemble_rb_population(
                state.particles, merge(projected, (; ancestor=Int32[]))
            )
            fs = state.particles.components
            indices = Array(fs.ancestor.data)
            xn = Array(fs.state.components.x.data)
            push!(population_x, xn)
            push!(population_ancestors, indices)
            @test xn[1, 1] == refs[t + 1]
            @test indices[1] == (t == 3 ? n : 1)
            @test Array(old_state.particles.components.state.components.x.data) == old_x
            next_beliefs = similar(beliefs)
            increments = zeros(Float32, n)
            for i in 1:n
                j = indices[i]
                dyn = GF.inner_dynamics(ReferenceInner(), t, xp[:, j], xn[:, i])
                next_beliefs[i], increments[i] = GF.kalman_update(
                    GF.kalman_predict(beliefs[j], dyn), obs, y
                )
            end
            incoming = threshold == 0 ? weights : zeros(Float32, n)
            raw = incoming .+ increments
            z = GF.logsumexp(raw)
            @test ll ≈ z - GF.logsumexp(incoming) atol=2.0f-5
            weights = raw .- z
            @test Array(GF.log_weights(state)) ≈ weights atol=2.0f-5
            @test Array(fs.state.components.z.components.μ.data) ≈
                hcat((b.μ for b in next_beliefs)...)
            @test Array(fs.state.components.z.components.Σ.data) ≈
                cat((b.Σ for b in next_beliefs)...; dims=3)
            xp, beliefs = xn, next_beliefs
        end
        # Trace every path independently on the host, including the nontrivial
        # conditioned ancestor at t=3. No production population is downloaded.
        for i in 1:n
            path = get_ancestry(history, i)
            a = i
            expected = zeros(Float32, length(refs))
            for t in 4:-1:1
                expected[t + 1] = population_x[t + 1][1, a]
                a = population_ancestors[t][a]
            end
            expected[1] = population_x[1][1, a]
            @test [only(Array(s.x)) for s in collect(path)] == expected
            sparse_path = get_ancestry(sparse.history, i)
            @test [only(Array(x)) for x in collect(sparse_path)] == expected
        end
        # Histories own batch buffers, including weights and ancestry.
        saved = Array(history.states[end].components.x.data)
        state.particles.components.state.components.x.data .= 100.0f0
        @test Array(history.states[end].components.x.data) == saved
        @test eltype(history.ancestors[end]) === Int32
        @test Array(storage) == reshape(refs, 1, :)

        # A single conditioned particle must return the complete prescribed path,
        # including time zero, through the ordinary shared CSMC entry point.
        one_particle = ConditionalSMC(RBPF(BF(1), KF()))
        ys = [CUDA.fill(0.15f0 * t, 1) for t in 1:4]
        trajectory, ll = GF._csmc_sample(CUDA.RNG(55), model, one_particle, ys, reference)
        @test [only(Array(x)) for x in collect(trajectory)] == refs
        @test isfinite(ll)
        @test_throws DimensionMismatch initialise(
            CUDA.RNG(1), model.prior, algo; ref_state=[CUDA.zeros(Float32, 2)]
        )
        @test_throws ArgumentError initialise(
            CUDA.RNG(1), model.prior, algo; ref_state=[CUDA.zeros(Float64, 1)]
        )
        @test_throws ArgumentError GF.step(
            CUDA.RNG(1), model, algo, 5, state, CUDA.zeros(Float32, 1); ref_state=reference
        )
    end
end
