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
N_particles = 50
N_iter = 5000
N_adapts = 1000

# HierarchicalSSM with inner drift b:
#   outer state — AR(1), sampled via particles
#   inner state — AR(1) with offset b + c_val * outer_state, marginalised via KF
#   observations — from inner state only
#
# All constant parts are pre-built so that PDMat constructors never appear in the
# Zygote trace — only getfield accesses, which Zygote handles natively.
const _fixed_ssm = let
    outer_prior = HomogeneousGaussianPrior(SVector{1}(0.0), PDMat(SMatrix{1,1}(σ₀²)))
    outer_dyn = HomogeneousLinearGaussianLatentDynamics(
        SMatrix{1,1}(a), SVector{1}(0.0), PDMat(SMatrix{1,1}(q²))
    )
    inner_prior = HomogeneousGaussianPrior(SVector{1}(0.0), PDMat(SMatrix{1,1}(σ₀²)))
    inner_dyn = GeneralisedFilters.GFTest.InnerDynamics(
        SMatrix{1,1}(a),
        SVector{1,Float64}([0.0]),
        SMatrix{1,1}(c_val),
        PDMat(SMatrix{1,1}(q²)),
    )
    inner_obs = HomogeneousLinearGaussianObservationProcess(
        SMatrix{1,1}(1.0), SVector{1}(0.0), PDMat(SMatrix{1,1}(r²))
    )
    HierarchicalSSM(outer_prior, outer_dyn, inner_prior, inner_dyn, inner_obs)
end

function build_ssm_rb(b)
    dyn = _fixed_ssm.inner_model.dyn
    new_inner_dyn = GeneralisedFilters.GFTest.InnerDynamics(
        dyn.A, SVector{1,Float64}(b), dyn.C, dyn.Q
    )
    return HierarchicalSSM(
        _fixed_ssm.outer_prior,
        _fixed_ssm.outer_dyn,
        _fixed_ssm.inner_model.prior,
        new_inner_dyn,
        _fixed_ssm.inner_model.obs,
    )
end

true_b = 1.5
_, _, _, _, ys = AbstractMCMC.sample(rng, build_ssm_rb([true_b]), T_len)

@model function drift_model_rb(ys)
    b ~ MvNormal([0.0], σ_b² * I)
    ssm = build_ssm_rb(b)
    x ~ SSMTrajectory(ssm, KF(), ys)
    return nothing
end

m = drift_model_rb(ys)
param_sampler = HMC(0.01, 10)
# adtype = ADTypes.AutoZygote()
adtype = ADTypes.AutoMooncake()
pg = ParticleGibbs(
    ConditionalSMC(RBPF(BF(N_particles), KF()), AncestorSampling()),
    param_sampler;
    adtype=adtype,
)

chain = AbstractMCMC.sample(
    rng, m, pg, N_iter; n_adapts=N_adapts, progress=true, chain_type=MCMCChains.Chains
)

@profview begin
    chain = AbstractMCMC.sample(
        rng, m, pg, N_iter; n_adapts=N_adapts, progress=true, chain_type=MCMCChains.Chains
    )
end

@benchmark AbstractMCMC.sample(
    rng, m, pg, N_iter; n_adapts=N_adapts, progress=true, chain_type=MCMCChains.Chains
)
