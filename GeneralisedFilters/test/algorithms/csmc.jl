"""Conditional particle Gibbs correctness against analytical smoothing."""

@testitem "CSMC Gaussian posterior and outer-only RB trajectories" begin
    using GeneralisedFilters, StableRNGs, StaticArrays, Statistics
    using AbstractMCMC: AbstractMCMC
    using LogExpFunctions: logsumexp
    const GF = GeneralisedFilters
    rng = StableRNG(8251)
    # Joint Gaussian model: z_t depends on BOTH x_{t-1} and x_t, and on time.
    # This catches stale ancestor beliefs and backward transition indexing errors.
    ax, az, u, v = 0.6, 0.7, 0.35, -0.2
    qx, qz, r = 0.4, 0.25, 0.3
    outer_prior = GaussianPrior(SVector(0.0), SMatrix{1,1}(0.8))
    outer_dyn = TimeVaryingDynamics(
        ((; t),) ->
            LinearGaussianDynamics(SMatrix{1,1}(ax), SVector(0.03t), SMatrix{1,1}(qx)),
    )
    inner_prior_fn = ((; x0),) -> GaussianPrior(SVector(0.3x0[1]), SMatrix{1,1}(0.5))
    inner_dyn_fn =
        ((; t, x_prev, x_new),) -> LinearGaussianDynamics(
            SMatrix{1,1}(az),
            SVector(u * x_prev[1] + v * x_new[1] + 0.02t),
            SMatrix{1,1}(qz),
        )
    inner_obs =
        ((; t, x),) ->
            LinearGaussianObservation(SMatrix{1,1}(1.0), SVector(0.4x[1]), SMatrix{1,1}(r))
    hier = StateSpaceModel(outer_prior, outer_dyn, inner_prior_fn, inner_dyn_fn, inner_obs)
    joint = StateSpaceModel(
        GaussianPrior(SVector(0.0, 0.0), @SMatrix [0.8 0.24; 0.24 0.572]),
        TimeVaryingDynamics(
            ((; t),) -> LinearGaussianDynamics(
                @SMatrix([ax 0.0; u+v * ax az]),
                SVector(0.03t, (v * 0.03 + 0.02) * t),
                @SMatrix([qx v*qx; v*qx qz+v^2 * qx])
            ),
        ),
        LinearGaussianObservation(@SMatrix([0.4 1.0]), SVector(0.0), SMatrix{1,1}(r)),
    )
    ys = [SVector(0.2), SVector(-0.4), SVector(0.7)]
    truth, _ = smooth(rng, joint, KalmanSmoother(), ys; t_smooth=1)
    for strategy in (NoRefreshment(), AncestorSampling(), BackwardSimulation())
        pf = RBPF(BF(12; resampler=Multinomial(), threshold=0.8), KF())
        sampler = ConditionalSMC(pf, strategy)
        xmeans = Float64[]
        zmeans = Float64[]
        xsq = Float64[]
        ref = nothing
        for i in 1:4200
            ref, ll = GF._csmc_sample(rng, hier, sampler, ys, ref)
            if i == 1
                @test isfinite(ll)
                @test ref isa ReferenceTrajectory
                @test ref[0] isa SVector{1,Float64}
                @test !(ref[1] isa GF.RBState)
            end
            if i > 200
                push!(xmeans, only(ref[1]))
                push!(xsq, only(ref[1])^2)
                inner, _ = smooth(
                    rng, condition_inner(hier, ref), KalmanSmoother(), ys; t_smooth=1
                )
                push!(zmeans, only(inner.μ))
            end
        end
        # Absolute tolerances avoid unstable relative checks when posterior means are near 0.
        @test mean(xmeans) ≈ truth.μ[1] atol = 0.055
        @test mean(zmeans) ≈ truth.μ[2] atol = 0.035
        @test mean(xsq) ≈ truth.Σ[1, 1] + truth.μ[1]^2 atol = 0.07
    end
    # Ancestor future-density differences equal a fresh full forward likelihood,
    # including a candidate-dependent initial inner distribution and transition offsets.
    suffix = [SVector(0.4), SVector(-0.3), SVector(0.6)]
    reference = ReferenceTrajectory(SVector(0.0), suffix)
    pf = RBPF(BF(8; resampler=Multinomial()), KF())
    likes = GF._compute_backward_likelihoods(
        rng, hier, pf, ys, reference, AncestorSampling()
    )
    @test isconcretetype(eltype(likes))
    # Mixed representations widen a cache without reading its uninitialised prefix.
    terminal = GF.InformationLikelihood([0.1], reshape([0.2], 1, 1))
    mixed = Vector{typeof(terminal)}(undef, 2)
    mixed[2] = terminal
    first_lik = GF.InformationLikelihood(SVector(0.3), SMatrix{1,1}(0.4))
    mixed = GF._store_backward_likelihood(mixed, 1, first_lik)
    @test mixed[1] === first_lik
    @test mixed[2] === terminal
    @test eltype(mixed) == Union{typeof(terminal),typeof(first_lik)}
    ref_as = GF.RBState(reference[1], likes[1])
    backward_scores = Float64[]
    forward_scores = Float64[]
    for x0 in (SVector(-0.8), SVector(0.7))
        prior = inner_prior(hier, x0)
        state0 = GF.RBState(x0, GaussianState(prior.μ0, prior.Σ0))
        push!(backward_scores, future_conditional_density(hier.dyn, pf, 1, state0, ref_as))
        path = ReferenceTrajectory(x0, suffix)
        push!(
            forward_scores,
            trajectory_logdensity(hier, KF(), path, ys) - logdensity(hier.prior.outer, x0),
        )
    end
    @test backward_scores[2] - backward_scores[1] ≈ forward_scores[2] - forward_scores[1] atol =
        1e-10

    # Ordinary CSMC also targets the same joint Gaussian posterior.
    for strategy in (NoRefreshment(), AncestorSampling(), BackwardSimulation())
        sampler = ConditionalSMC(BF(16; resampler=Multinomial()), strategy)
        ref = nothing
        draws = SVector{2,Float64}[]
        for i in 1:2200
            ref, ll = GF._csmc_sample(rng, joint, sampler, ys, ref)
            i > 200 && push!(draws, ref[1])
        end
        @test mean(draws) ≈ truth.μ atol = 0.08
    end
    auxiliary = AuxiliaryParticleFilter(BF(8; resampler=Multinomial()), MeanPredictive())
    aux_sampler = ConditionalSMC(auxiliary)
    aux_ref, aux_ll = GF._csmc_sample(rng, joint, aux_sampler, ys, nothing)
    aux_ref, aux_ll = GF._csmc_sample(rng, joint, aux_sampler, ys, aux_ref)
    @test isfinite(aux_ll)
    @test length(aux_ref) == 4
    @test_throws ArgumentError GF._csmc_sample(
        rng, joint, ConditionalSMC(auxiliary, AncestorSampling()), ys, aux_ref
    )
    @test_throws ArgumentError GF._csmc_sample(
        rng, joint, ConditionalSMC(auxiliary, BackwardSimulation()), ys, aux_ref
    )
    # Public standalone chain entry points.
    cm = CSMCModel(hier, ys)
    sampler = ConditionalSMC(RBPF(BF(8; resampler=Multinomial()), KF()), AncestorSampling())
    transition, state = AbstractMCMC.step(rng, cm, sampler)
    @test state isa CSMCState
    _, next_state = AbstractMCMC.step(rng, cm, sampler, state)
    @test next_state.trajectory[0] isa SVector
    @test length(next_state.trajectory) == 4
    @test_throws ArgumentError GF._csmc_sample(
        rng, hier, ConditionalSMC(RBPF(BF(8), KF())), ys, nothing
    )
    @test_throws ArgumentError GF._csmc_sample(
        rng, hier, sampler, SVector{1,Float64}[], nothing
    )
    @test_throws DimensionMismatch GF._csmc_sample(
        rng, hier, sampler, ys, ReferenceTrajectory(SVector(0.0), [SVector(0.0)])
    )
    @test_throws ArgumentError GF._csmc_sample(
        rng,
        hier,
        ConditionalSMC(
            RBPF(BF(8; resampler=Multinomial()), KalmanFilter(; repair=Jitter(1e-8))),
            AncestorSampling(),
        ),
        ys,
        nothing,
    )
    for strategy in (AncestorSampling, BackwardSimulation),
        bp in (
            BackwardInformationPredictor(; initial_jitter=1e-8),
            BackwardInformationPredictor(; jitter=1e-8),
        )

        @test_throws ArgumentError GF._csmc_sample(
            rng,
            hier,
            ConditionalSMC(RBPF(BF(8; resampler=Multinomial()), KF()), strategy(bp)),
            ys,
            nothing,
        )
    end
    # With one particle and a pinned path, the evidence is exactly the conditional
    # inner likelihood. Changing the model must recompute every inner belief.
    changed = StateSpaceModel(
        hier.prior,
        hier.dyn,
        HierarchicalObservation(
            ((; t, x),) -> LinearGaussianObservation(
                SMatrix{1,1}(1.0), SVector(0.4x[1]), SMatrix{1,1}(1.2)
            ),
        ),
    )
    for strategy in (NoRefreshment(), AncestorSampling(), BackwardSimulation())
        one = ConditionalSMC(RBPF(BF(1; resampler=Multinomial()), KF()), strategy)
        path, ll = GF._csmc_sample(rng, changed, one, ys, reference)
        @test collect(path) == collect(reference)
        @test ll ≈ inner_loglikelihood(KF(), changed, reference, ys) atol = 1e-12
        short_ref = ReferenceTrajectory(reference[0], [reference[1]])
        short_path, short_ll = GF._csmc_sample(rng, changed, one, ys[1:1], short_ref)
        @test collect(short_path) == collect(short_ref)
        @test short_ll ≈ inner_loglikelihood(KF(), changed, short_ref, ys[1:1]) atol = 1e-12
    end
    explicit = ConditionalSMC(
        RBPF(BF(8; resampler=Multinomial()), KF()),
        BackwardSimulation(BackwardInformationPredictor()),
    )
    ref, _ = GF._csmc_sample(rng, hier, explicit, ys, nothing)
    @test length(ref) == 4
end

@testitem "Discrete CSMC exact enumerated posterior" begin
    using GeneralisedFilters, StableRNGs, Statistics
    using Distributions: Normal, pdf
    const GF = GeneralisedFilters
    rng = StableRNG(539)
    outer_p = DiscretePrior([0.6, 0.4])
    outer_d = DiscreteDynamics([0.8 0.2; 0.3 0.7])
    inner_p = ((; x0),) -> DiscretePrior(x0 == 1 ? [0.8, 0.2] : [0.25, 0.75])
    inner_d =
        ((; t, x_prev, x_new),) ->
            DiscreteDynamics(x_prev == x_new ? [0.9 0.1; 0.2 0.8] : [0.4 0.6; 0.65 0.35])
    inner_o =
        ((; t, x),) -> DistributionObservation((_, z) -> Normal(1.2x + 0.8z + 0.1t, 0.7))
    model = StateSpaceModel(outer_p, outer_d, inner_p, inner_d, inner_o)
    ys = [2.4, 3.0, 2.0]
    paths = collect(Iterators.product(fill(1:2, 4)...))
    probs = map(paths) do path
        # Explicit joint sum over inner trajectories is independent of filter implementation.
        sum(Iterators.product(fill(1:2, 4)...)) do zs
            p = outer_p.α0[path[1]] * inner_p((; x0=path[1])).α0[zs[1]]
            for t in 1:3
                p *= outer_d.P[path[t], path[t + 1]]
                p *= inner_d((; t, x_prev=path[t], x_new=path[t + 1])).P[zs[t], zs[t + 1]]
                p *= pdf(Normal(1.2path[t + 1] + 0.8zs[t + 1] + 0.1t, 0.7), ys[t])
            end
            p
        end
    end
    truth = sum(p * (path[2] == 1) for (p, path) in zip(probs, paths)) / sum(probs)
    for strategy in (NoRefreshment(), AncestorSampling(), BackwardSimulation())
        sampler = ConditionalSMC(RBPF(BF(8; resampler=Multinomial()), DF()), strategy)
        ref = nothing
        count = 0
        for i in 1:5200
            ref, ll = GF._csmc_sample(rng, model, sampler, ys, ref)
            i > 200 && (count += ref[1] == 1)
        end
        @test count / 5000 ≈ truth atol = 0.035
        @test ref[0] isa Int
    end
end
