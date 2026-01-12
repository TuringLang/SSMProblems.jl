using Distributions
using LinearAlgebra
using Random
using SSMProblems
using GeneralisedFilters
using ProgressMeter
using StatsBase
using Plots
using MCMCDiagnosticTools
using Profile
using StaticArrays

# Load local research module
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using GaelResearch

const SEED = 12345
const Dx = 1
const Dy = 1
const K = 10
const T = Float64
const N_particles = 1000
const N_burnin = 100
const N_sample = 1000
const TUNE_PARTICLES = true


@enum samplers PMMH_TYPE PGIBBS_TYPE EHMM_TYPE
const DEFAULT_SAMPLER = PMMH_TYPE

function main(sampler_type::samplers = DEFAULT_SAMPLER)
    rng = MersenneTwister(SEED)

    μ = 0.0; σ2 = 1.0
    b_prior = MvNormal(zeros(T, Dx) .+ μ, σ2 * I)
    b_true = rand(rng, b_prior)
    println("True b: ", b_true)

    # Generate model matrices/vectors
    μ0 = @SVector rand(rng, T, Dx)
    Σ0 = SMatrix{Dx, Dx}(rand_cov(rng, T, Dx))
    A = @SMatrix rand(rng, T, Dx, Dx)
    Q = SMatrix{Dx, Dx}(rand_cov(rng, T, Dx))
    H = @SMatrix rand(rng, T, Dy, Dx)
    c = @SVector rand(rng, T, Dy)
    R = SMatrix{Dy, Dy}(rand_cov(rng, T, Dy))

    # Define full model and sample observations
    full_model = create_homogeneous_linear_gaussian_model(μ0, Distributions.PDMat(Σ0), A, b_true, Distributions.PDMat(Q), H, c, Distributions.PDMat(R))
    _, xs, ys = sample(rng, full_model, K)

    # Define augmented dynamics
    μ0_aug = vcat(μ0, b_prior.μ) # SVector concatenation
    # Construct block diagonal matrices manually or via helper for StaticArrays if needed, 
    # but for simplicity and since these are init params, we can keep them standard or make them static.
    # Ideally, keep everything static.
    
    # helper for block diag
    z_Dx = @SMatrix zeros(T, Dx, Dx)
    z_Dy_Dx = @SMatrix zeros(T, Dy, Dx)
    
    # We need to be careful with block construction for SMatrix to ensure types are correct
    # A_aug construction:
    # A  I
    # 0  I
    
    A_aug = SMatrix{2*Dx, 2*Dx}([
        A I;
        z_Dx I
    ])
    
    Σ0_aug = SMatrix{2*Dx, 2*Dx}([
        Σ0 z_Dx;
        z_Dx b_prior.Σ
    ])
    
    b_aug = @SVector zeros(T, 2 * Dx)
    
    Q_aug = SMatrix{2*Dx, 2*Dx}([
        Q z_Dx;
        z_Dx z_Dx
    ])

    H_aug = SMatrix{Dy, 2*Dx}([H z_Dy_Dx])

    # Create augmented model
    aug_model = create_homogeneous_linear_gaussian_model(
        μ0_aug, Distributions.PDMat(Σ0_aug), A_aug, b_aug, Distributions.PDMat(Q_aug), H_aug, c, Distributions.PDMat(R)
    )
    state, _ = GeneralisedFilters.filter(rng, aug_model, KalmanFilter(), ys)
    println("Ground truth posterior mean: ", state.μ[(Dx+1):end])

    function model_builder(θ)
        # Ensure θ is SVector to prevent type instability in the model
        θ_static = SVector{Dx}(θ)
        return create_homogeneous_linear_gaussian_model(
            μ0, Distributions.PDMat(Σ0), A, θ_static, Distributions.PDMat(Q), H, c, Distributions.PDMat(R)
        )
    end

    Qinv = inv(Q)
    Σ_prior_inv = inv(b_prior.Σ)
    μ_prior = b_prior.μ # Capture value
    
    function b_sampler(ref_traj, rng, xs)
        # compute residuals r_t = x_t - A * x_{t-1} for t=2..T
        residuals = [ref_traj[t] - A * ref_traj[t-1] for t in (firstindex(ref_traj) + 1):lastindex(ref_traj)]
        n = length(residuals)
        sum_r = reduce(+, residuals)

        Σ_post = inv(n * Qinv + Σ_prior_inv)
        μ_post = Σ_post * (Qinv * sum_r + Σ_prior_inv * μ_prior)

        return SVector{Dx}(rand(rng, MvNormal(vec(μ_post), Symmetric(Σ_post))))
    end

    # Setup AbstractMCMC model
    model = ParameterisedSSM(model_builder, b_prior)
    bf = BF(N_particles; threshold=1.0)

    println("Starting sampling using sampler type: ", sampler_type)

    sampler = if sampler_type == PMMH_TYPE
        println("Estimating log-likelihood variance...")
        b_curr = b_prior.μ
        m_curr = model_builder(b_curr)
        
        N_est, V = estimate_particle_count(rng, m_curr, ys, N -> BF(N; threshold=1.0); initial_N=N_particles)
        println("Log-likelihood variance at N=$N_particles: ", V)
        if TUNE_PARTICLES println("Estimated particles for variance=1.0: ", N_est) end
        
        N_run = TUNE_PARTICLES ? N_est : N_particles
        bf_tuned = BF(N_run; threshold=1.0)
        PMMH(bf_tuned; d=length(b_prior), adapt_end=N_burnin)
    elseif sampler_type == PGIBBS_TYPE
        PGibbs(bf, b_sampler)

    elseif sampler_type == EHMM_TYPE
        EHMM(bf, b_sampler, 10)
    end

    samples = @profile sample(rng, model, sampler, ys; n_samples=N_sample, n_burnin=N_burnin, init_θ=SVector{Dx}(b_prior.μ))
    # Profile.print(format=:flat, sortedby=:count, mincount=50)

    println("Posterior mean: ", mean(samples))
    println("Effective sample size: ", ess(stack(samples)'))
end

main()