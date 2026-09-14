"""The APF wrapper must preserve the square-root conditional target in Turing."""
@testitem "Turing APF square-root RBPG with both AD modes" begin
    using GeneralisedFilters, StaticArrays, StableRNGs, ADTypes
    using ForwardDiff, Mooncake, Turing, MCMCChains
    using AbstractMCMC: AbstractMCMC
    using AdvancedHMC: NUTS

    function build_model(θ)
        outer_prior = GaussianPrior(SVector(0.0), SMatrix{1,1}(1.0))
        outer_dyn = LinearGaussianDynamics(
            SMatrix{1,1}(0.7), SVector(0.0), SMatrix{1,1}(0.3)
        )
        inner_prior = ((; x0),) -> GaussianPrior(SVector(0.2x0[1]), SMatrix{1,1}(0.6))
        inner_dyn =
            ((; t, x_prev, x_new),) -> LinearGaussianDynamics(
                SMatrix{1,1}(0.5),
                SVector(0.2x_prev[1] + 0.3x_new[1]),
                CovarianceFactor(SMatrix{1,1}(exp(θ))),
            )
        inner_obs =
            ((; t, x),) -> LinearGaussianObservation(
                SMatrix{1,1}(1.0), SVector(0.1x[1]), SMatrix{1,1}(0.5)
            )
        return StateSpaceModel(outer_prior, outer_dyn, inner_prior, inner_dyn, inner_obs)
    end
    ys = [SVector(0.1), SVector(-0.4), SVector(0.6)]
    @model function inference_model(ys)
        θ ~ Normal(-0.5, 0.2)
        return x ~ SSMTrajectory(build_model(θ), SRKF(), ys)
    end
    for (adtype, refreshment) in
        ((AutoForwardDiff(), AncestorSampling()), (AutoMooncake(), BackwardSimulation()))
        pf = AuxiliaryParticleFilter(
            RBPF(BF(8; resampler=GeneralisedFilters.Multinomial(), threshold=0.8), SRKF()),
            GeneralisedFilters.MeanPredictive(),
        )
        pg = ParticleGibbs(ConditionalSMC(pf, refreshment), NUTS(0.8); adtype)
        chain = AbstractMCMC.sample(
            StableRNG(819),
            inference_model(ys),
            pg,
            8;
            n_adapts=4,
            progress=false,
            chain_type=MCMCChains.Chains,
        )
        @test size(chain, 1) == 8
        @test all(isfinite, Array(chain[:θ]))
    end
end
