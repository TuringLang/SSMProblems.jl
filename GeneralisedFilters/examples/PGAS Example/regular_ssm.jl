using GeneralisedFilters
using AbstractMCMC: AbstractMCMC
using AdvancedHMC: NUTS
using MCMCChains: MCMCChains
using Turing: @model
using Distributions
using PDMats
using LinearAlgebra
using Random
using Statistics
using SSMProblems
using StaticArrays
using Zygote

rng = MersenneTwister(1234)

a = 0.8
q² = 0.1
r² = 0.5
σ₀² = 1.0
σ_b² = 4.0
T_len = 100
N_particles = 50
N_iter = 5000
N_adapts = 1000

function build_ssm_reg(drift)
    return create_homogeneous_linear_gaussian_model(
        SVector{1}(0.0),
        PDMat(SMatrix{1,1}(σ₀²)),
        SMatrix{1,1}(a),
        SVector{1}(only(drift)),
        PDMat(SMatrix{1,1}(q²)),
        SMatrix{1,1}(1.0),
        SVector{1}(0.0),
        PDMat(SMatrix{1,1}(r²)),
    )
end

true_b = 1.5
true_ssm = build_ssm_reg([true_b])
_, _, ys = SSMProblems.sample(rng, true_ssm, T_len)

@model function drift_model_reg(ys)
    b ~ MvNormal([0.0], σ_b² * I)
    ssm = build_ssm_reg(b)
    x ~ SSMTrajectory(ssm, ys)
    return nothing
end

m = drift_model_reg(ys)
pg = ParticleGibbs(ConditionalSMC(BF(N_particles), AncestorSampling()), NUTS(0.8))

chain = AbstractMCMC.sample(
    rng, m, pg, N_iter; n_adapts=N_adapts, progress=false, chain_type=MCMCChains.Chains
)

display(
    @benchmark AbstractMCMC.sample(
        rng, m, pg, N_iter; n_adapts=N_adapts, progress=false, chain_type=MCMCChains.Chains
    )
)

@profview AbstractMCMC.sample(
    rng, m, pg, N_iter; n_adapts=N_adapts, progress=false, chain_type=MCMCChains.Chains
)
