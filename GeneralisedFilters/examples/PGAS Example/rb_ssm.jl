using GeneralisedFilters, AbstractMCMC, AdvancedHMC, ADTypes, MCMCChains
using Turing: @model
using Distributions, LinearAlgebra, Random, StaticArrays, ForwardDiff, Mooncake

# StaticArray RBPG with a positive parameter shared by both processes.
function build_rb(b, q)
    return StateSpaceModel(
        GaussianPrior(SVector(0.0), SMatrix{1,1}(1.0)),
        LinearGaussianDynamics(SMatrix{1,1}(0.8), SVector(0.0), SMatrix{1,1}(q)),
        GaussianPrior(SVector(0.0), SMatrix{1,1}(1.0)),
        ctx -> LinearGaussianDynamics(
            SMatrix{1,1}(0.8), SVector(b + 0.5ctx.x_new[1]), SMatrix{1,1}(q)
        ),
        LinearGaussianObservation(SMatrix{1,1}(1.0), SVector(0.0), SMatrix{1,1}(0.5)),
    )
end

@model function rb_model(ys)
    b ~ Normal(0.0, 2.0)
    q ~ LogNormal(-2.0, 0.5)
    return x ~ SSMTrajectory(build_rb(b, q), KF(), ys)
end

function run_rb_example(; backend=AutoForwardDiff(), iterations=1000, adaptation=200)
    rng = MersenneTwister(1234)
    _, _, ys = simulate(rng, build_rb(1.5, 0.1), 30)
    csmc = ConditionalSMC(
        RBPF(BF(50; resampler=GeneralisedFilters.Multinomial()), KF()), AncestorSampling()
    )
    pg = ParticleGibbs(csmc, AdvancedHMC.NUTS(0.8); adtype=backend)
    return AbstractMCMC.sample(
        rng,
        rb_model(ys),
        pg,
        iterations;
        n_adapts=adaptation,
        progress=false,
        chain_type=MCMCChains.Chains,
    )
end

# Reverse mode: run_rb_example(backend=AutoMooncake(; config=nothing)).
if abspath(PROGRAM_FILE) == @__FILE__
    display(run_rb_example())
end
