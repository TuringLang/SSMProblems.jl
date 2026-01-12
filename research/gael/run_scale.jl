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

# Helper functions and types (Must be top level)
struct ZeroLikelihoodModel end
GeneralisedFilters.filter(::AbstractRNG, ::ZeroLikelihoodModel, ::GeneralisedFilters.AbstractFilter, ::AbstractVector) = (nothing, -Inf)


function main(sampler_type::samplers = DEFAULT_SAMPLER)
    rng = MersenneTwister(SEED)

    α = 3.0; β = 2.0
    σ2_prior = InverseGamma(α, β)
    σ2_true = rand(rng, σ2_prior)
    println("True σ2: ", σ2_true)

    # Generate model matrices/vectors
    μ0 = @SVector rand(rng, T, Dx)
    Σ0 = SMatrix{Dx, Dx}(rand_cov(rng, T, Dx))
    A = @SMatrix rand(rng, T, Dx, Dx)
    b = @SVector rand(rng, T, Dx)
    Q = SMatrix{Dx, Dx}(rand_cov(rng, T, Dx))
    H = @SMatrix rand(rng, T, Dy, Dx)
    c = @SVector rand(rng, T, Dy)
    R = SMatrix{Dy, Dy}(rand_cov(rng, T, Dy))

    # Define full model and sample observations
    full_model = create_homogeneous_linear_gaussian_model(μ0, Distributions.PDMat(Σ0), A, b, Distributions.PDMat(σ2_true .* Q), H, c, Distributions.PDMat(σ2_true .* R))
    _, xs, ys = sample(rng, full_model, K)

    model_init = create_homogeneous_linear_gaussian_model(μ0, Distributions.PDMat(Σ0), A, b, Distributions.PDMat(Q), H, c, Distributions.PDMat(R))
    cb = GeneralisedFilters.StateCallback(nothing, nothing)
    _, _ = GeneralisedFilters.filter(rng, model_init, KalmanFilter(), ys; callback=cb)

    function update_closed_params(α, β, ys, cb, model, Dy)
        α_closed = α
        β_closed = β

        for t in eachindex(ys)
            μ_pred, Σ_pred = mean(cb.proposed_states[t]), cov(cb.proposed_states[t])

            ŷ = H * μ_pred + c
            S = H * Σ_pred * H' + R
            e = ys[t] - ŷ

            α_closed += Dy / 2
            β_closed += 0.5 * dot(e, S \ e)
        end

        return α_closed, β_closed
    end

    α_closed, β_closed = update_closed_params(α, β, ys, cb, model_init, Dy)
    println("Ground truth posterior mean: ", β_closed / (α_closed - 1))


    function model_builder(θ)
        if θ[1] .<= 0.0
            return ZeroLikelihoodModel()
        end
        return create_homogeneous_linear_gaussian_model(
            μ0, Distributions.PDMat(Σ0), A, b, Distributions.PDMat(θ[1] .* Q), H, c, Distributions.PDMat(θ[1] .* R)
        )
    end

    invQ = inv(Q)
    invR = inv(R)
    
    function q_sampler(ref_traj, rng, xs)
        # Residuals: process x_t - (A x_{t-1} + b), observation y_t - (H x_t + c)
        proc_res = [ref_traj[t] .- (A * ref_traj[t - 1] .+ b) for t in (firstindex(ref_traj) + 1):lastindex(ref_traj)]
        obs_res = [ys[t] .- (H * ref_traj[t] .+ c) for t in firstindex(ys):lastindex(ys)]

        # Sum of Mahalanobis distances with base covariances (Var = θ * Q/R)
        # proc_res elements are SVector, use dot directly
        proc_ss = sum(r -> dot(r, invQ * r), proc_res)
        obs_ss = sum(r -> dot(r, invR * r), obs_res)
        ss = proc_ss + obs_ss

        # Total degrees of freedom contributed by all residual components
        n = length(proc_res) * Dx + length(obs_res) * Dy

        α_post = α + n / 2
        β_post = β + ss / 2

        return SVector{1}(rand(rng, InverseGamma(α_post, β_post)))
    end

    # Setup AbstractMCMC model
    prior_logpdf(θ) = (θ[1] <= 0) ? -Inf : logpdf(σ2_prior, θ[1])
    model = ParameterisedSSM(model_builder, prior_logpdf)
    bf = BF(N_particles; threshold=1.0)

    println("Starting sampling using sampler type: ", sampler_type)

    θ_curr = [β / (α - 1)]

    sampler = if sampler_type == PMMH_TYPE
        println("Estimating log-likelihood variance...")
        m_curr = model_builder(θ_curr)
        
        N_est, V = estimate_particle_count(rng, m_curr, ys, N -> BF(N; threshold=1.0); initial_N=N_particles)
        println("Log-likelihood variance at N=$N_particles: ", V)
        if TUNE_PARTICLES println("Estimated particles for variance=1.0: ", N_est) end
        
        N_run = TUNE_PARTICLES ? N_est : N_particles
        bf_tuned = BF(N_run; threshold=1.0)

        PMMH(bf_tuned; d=1, adapt_end=N_burnin)
    elseif sampler_type == PGIBBS_TYPE
        PGibbs(bf, q_sampler)
    elseif sampler_type == EHMM_TYPE
        EHMM(bf, q_sampler, 10)
    end

    samples = @profile sample(rng, model, sampler, ys; n_samples=N_sample, n_burnin=N_burnin, init_θ=SVector{1}(θ_curr))
    #Profile.print(format=:flat, sortedby=:count, mincount=50)

    println("Posterior mean: ", mean(samples))
    println("Effective sample size: ", ess(stack(samples)'))
end

main()

