@testitem "APF refreshment uses filtering weights and selected ancestor correction" begin
    using GeneralisedFilters, StableRNGs
    using LogExpFunctions: softmax
    const GF = GeneralisedFilters
    rng = StableRNG(751)
    model = StateSpaceModel(
        DiscretePrior([0.6, 0.4]),
        DiscreteDynamics([0.8 0.2; 0.3 0.7]),
        DistributionObservation((t, x) -> GF.Distributions.Normal(x, 0.7)),
    )
    pf = BF(3; resampler=Multinomial())
    apf = AuxiliaryParticleFilter(pf, ModePredictive())
    particles = [
        GF.Particle(1, log(0.2), 1),
        GF.Particle(2, log(0.3), 2),
        GF.Particle(1, log(0.5), 3),
    ]
    state = GF.ParticleDistribution(particles, 0.0)
    # Very different auxiliary and filtering laws make double-counting visible.
    eta = [3.0, -4.0, 0.5]
    for ref_idx in 1:3
        aux = GF.AuxiliaryResampler(GF.resampler(apf), eta)
        selected = GF.resample(rng, aux, state; ref_state=[1], ref_idx)
        @test selected.particles[1].ancestor == ref_idx
        @test selected.particles[1].log_w == -eta[ref_idx]
        @test selected.particles[1].state == state.particles[ref_idx].state
        scores = [ancestor_weight(p, model.dyn, apf, 1, 2) for p in selected.particles]
        expected = [
            p.log_w + logdensity(model.dyn, 1, p.state, 2) for p in selected.particles
        ]
        @test softmax(scores) ≈ softmax(expected)
    end
    # With one particle, inverse lookahead and normalizer cancel exactly.
    ys = [0.2, 1.8, 0.6]
    reference = ReferenceTrajectory(1, [1, 2, 1])
    expected_ll = sum(logdensity(model.obs, t, reference[t], ys[t]) for t in 1:3)
    for strategy in (AncestorSampling(), BackwardSimulation()), threshold in (0.0, 1.0)
        one = AuxiliaryParticleFilter(
            BF(1; resampler=Multinomial(), threshold), ModePredictive()
        )
        path, ll = GF._csmc_sample(rng, model, ConditionalSMC(one, strategy), ys, reference)
        @test collect(path) == collect(reference)
        @test ll ≈ expected_ll
    end
end

@testitem "APF AS and BS target exact ordinary and RB discrete posteriors" begin
    using GeneralisedFilters, StableRNGs
    using Distributions: Normal, pdf, Categorical
    const GF = GeneralisedFilters
    struct APFRefreshmentProposal <: AbstractProposal end
    GF.distribution(::APFRefreshmentProposal, t::Integer, state, observation) =
        Categorical([0.55, 0.45])
    rng = StableRNG(94513)
    outer_p = DiscretePrior([0.6, 0.4])
    outer_d = DiscreteDynamics([0.8 0.2; 0.3 0.7])
    inner_p = ((; x0),) -> DiscretePrior(x0 == 1 ? [0.8, 0.2] : [0.25, 0.75])
    inner_d =
        ((; t, x_prev, x_new),) ->
            DiscreteDynamics(x_prev == x_new ? [0.9 0.1; 0.2 0.8] : [0.4 0.6; 0.65 0.35])
    inner_o =
        ((; t, x),) -> DistributionObservation((_, z) -> Normal(1.2x + 0.8z + 0.1t, 0.7))
    rb_model = StateSpaceModel(outer_p, outer_d, inner_p, inner_d, inner_o)
    ordinary_model = StateSpaceModel(
        outer_p, outer_d, DistributionObservation((t, x) -> Normal(1.2x + 0.1t, 0.7))
    )
    ys = [2.4, 3.0, 2.0]
    paths = collect(Iterators.product(fill(1:2, 4)...))
    for rb in (false, true)
        model = rb ? rb_model : ordinary_model
        probs = map(paths) do path
            outer = outer_p.α0[path[1]] * prod(outer_d.P[path[t], path[t + 1]] for t in 1:3)
            if rb
                inner = sum(Iterators.product(fill(1:2, 4)...)) do zs
                    p = inner_p((; x0=path[1])).α0[zs[1]]
                    for t in 1:3
                        p *= inner_d((; t, x_prev=path[t], x_new=path[t + 1])).P[
                            zs[t], zs[t + 1]
                        ]
                        p *= pdf(Normal(1.2path[t + 1] + 0.8zs[t + 1] + 0.1t, 0.7), ys[t])
                    end
                    p
                end
                outer * inner
            else
                outer * prod(pdf(Normal(1.2path[t + 1] + 0.1t, 0.7), ys[t]) for t in 1:3)
            end
        end
        truth = [
            sum(p * (path[t + 1] == 1) for (p, path) in zip(probs, paths)) / sum(probs) for
            t in 0:3
        ]
        # Include no resampling, adaptive resampling and every-step resampling.
        for strategy in (AncestorSampling(), BackwardSimulation()),
            threshold in (0.0, 0.75, 1.0),
            guided in (false, true)

            base = if guided
                PF(12, APFRefreshmentProposal(); resampler=Multinomial(), threshold)
            else
                BF(12; resampler=Multinomial(), threshold)
            end
            pf = AuxiliaryParticleFilter(rb ? RBPF(base, DF()) : base, ModePredictive())
            sampler = ConditionalSMC(pf, strategy)
            reference = nothing
            counts = zeros(4)
            for i in 1:5200
                reference, ll = GF._csmc_sample(rng, model, sampler, ys, reference)
                if i > 200
                    for t in 0:3
                        counts[t + 1] += reference[t] == 1
                    end
                end
            end
            @test counts / 5000 ≈ truth atol = 0.06
            @test reference[0] isa Int
        end
    end
end
