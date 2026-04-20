using GeneralisedFilters
using AbstractMCMC: AbstractMCMC
using AdvancedHMC
using ADTypes: ADTypes
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
using Mooncake

rng = MersenneTwister(1234)

a = 0.8
c_val = 0.5   # coupling: inner offset = b + c_val * outer_state
q² = 0.1
r² = 0.5
σ₀² = 1.0
σ_b² = 4.0
T_len = 100
N_particles = 10
N_iter = 1000
N_adapts = 500

# HierarchicalSSM with inner drift b:
#   outer state — AR(1), sampled via particles
#   inner state — AR(1) with offset b + c_val * outer_state, marginalised via KF
#   observations — from inner state only
#
# The factory pre-builds every component that does not depend on b and closes
# over them, so each call to the returned builder only allocates the single
# b-dependent InnerDynamics. All captured values have concrete static types,
# which keeps the closure type stable.
function make_ssm_builder(; a, q², c_val, r², σ₀²)
    outer_prior = HomogeneousGaussianPrior(SVector{1}(0.0), PDMat(SMatrix{1,1}(σ₀²)))
    outer_dyn = HomogeneousLinearGaussianLatentDynamics(
        SMatrix{1,1}(a), SVector{1}(0.0), PDMat(SMatrix{1,1}(q²))
    )
    inner_prior = HomogeneousGaussianPrior(SVector{1}(0.0), PDMat(SMatrix{1,1}(σ₀²)))
    A_inner = SMatrix{1,1}(a)
    C = SMatrix{1,1}(c_val)
    Q_inner = PDMat(SMatrix{1,1}(q²))
    inner_obs = HomogeneousLinearGaussianObservationProcess(
        SMatrix{1,1}(1.0), SVector{1}(0.0), PDMat(SMatrix{1,1}(r²))
    )
    return function (b)
        inner_dyn = GeneralisedFilters.GFTest.InnerDynamics(
            A_inner, SVector{1,Float64}(b), C, Q_inner
        )
        return HierarchicalSSM(outer_prior, outer_dyn, inner_prior, inner_dyn, inner_obs)
    end
end

build_ssm_rb = make_ssm_builder(; a=a, q²=q², c_val=c_val, r²=r², σ₀²=σ₀²)

true_b = 1.5
_, _, _, _, ys = AbstractMCMC.sample(rng, build_ssm_rb([true_b]), T_len)

## Turing path #################################################################################

@model function drift_model_rb(ys, build_ssm_rb, σ_b²)
    b ~ MvNormal([0.0], σ_b² * I)
    ssm = build_ssm_rb(b)
    x ~ SSMTrajectory(ssm, KF(), ys)
    return nothing
end

m = drift_model_rb(ys, build_ssm_rb, σ_b²)
param_sampler = AdvancedHMC.HMC(0.01, 10)
# adtype = ADTypes.AutoZygote()
adtype = ADTypes.AutoMooncake()
# adtype = ADTypes.AutoForwardDiff()
pg = ParticleGibbs(
    ConditionalSMC(RBPF(BF(N_particles), KF()), AncestorSampling()),
    # ConditionalSMC(RBPF(BF(N_particles), KF()), BackwardSimulation()),
    param_sampler;
    adtype=adtype,
)

chain = AbstractMCMC.sample(
    rng, m, pg, N_iter; n_adapts=N_adapts, progress=true, chain_type=MCMCChains.Chains
)

# using Turing
# nuts = Turing.NUTS(N_adapts, 0.8; adtype=AutoMooncake(; config=nothing))
# AbstractMCMC.sample(
#     rng, m, nuts, N_iter; n_adapts=N_adapts, progress=true, chain_type=MCMCChains.Chains
# )

# display(
#     @benchmark AbstractMCMC.sample(
#         rng, m, pg, N_iter; n_adapts=N_adapts, progress=true, chain_type=MCMCChains.Chains
#     )
# )

@profview AbstractMCMC.sample(
    rng, m, pg, N_iter * 10; n_adapts=N_adapts, progress=true, chain_type=MCMCChains.Chains
)

# using Test

# ssm = build_ssm_rb([1.5])
# csmc = ConditionalSMC(RBPF(BF(N_particles), KF()), BackwardSimulation())

# # # Check return type is inferred
# traj, _ = GeneralisedFilters._csmc_sample(rng, ssm, csmc, ys, nothing)
# @profview for _ in 1:10000
#     GeneralisedFilters._csmc_sample(rng, ssm, csmc, ys, traj)
# end
