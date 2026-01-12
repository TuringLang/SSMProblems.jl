module Algorithms

using AbstractMCMC
using Random
using Distributions
using LinearAlgebra
using GeneralisedFilters
using SSMProblems
using ProgressMeter
using StatsBase
using LogExpFunctions
using ..Models
using ..Utils


using OffsetArrays
using StaticArrays

export PMMH, PGibbs, EHMM

# Helper for Cholesky with jitter
function _chol_with_jitter(Σ::AbstractMatrix; jitter=1e-8, max_tries::Int=8)
    d = size(Σ, 1)
    Σs = Symmetric(Σ)
    ϵ = jitter
    for _ in 1:max_tries
        try
            return cholesky(Σs + ϵ * I, check=true).L
        catch
            ϵ *= 10
        end
    end
    return cholesky(Σs + ϵ * I, check=false).L
end

## PMMH ########################################################################

struct PMMH{F<:GeneralisedFilters.AbstractFilter} <: AbstractMCMC.AbstractSampler
    filter_algo::F
    Σ_prop::AbstractMatrix
    scale::Float64
    adapt::Bool
    adapt_start::Int
    adapt_end::Int
    adapt_interval::Int
    jitter::Float64
end

function PMMH(filter_algo; 
    Σ0 = nothing, 
    scale = nothing, 
    adapt = true, 
    adapt_start = 0, 
    adapt_end = 1000, 
    adapt_interval = 1,
    jitter = 1e-8,
    d = 1
)
    Σ_prop = Σ0 === nothing ? 1e-2 * I(d) : Matrix(Σ0)
    s = scale === nothing ? (2.38^2) / d : float(scale)
    return PMMH(filter_algo, Σ_prop, s, adapt, adapt_start, adapt_end, adapt_interval, jitter)
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::ParameterisedSSM,
    sampler::PMMH,
    observations::AbstractVector;
    n_samples::Int,
    n_burnin::Int = 0,
    init_θ = nothing,
    kwargs...
)
    θ = init_θ === nothing ? rand(rng, model.prior) : init_θ
    d = length(θ)
    
    samples = Vector{typeof(θ)}(undef, n_samples)
    
    logprior(θ) = model.prior isa Function ? model.prior(θ) : logpdf(model.prior, θ)
    
    # Online stats for adaptation
    n_stats = 0
    μ_acc = zeros(eltype(θ), d)
    C_acc = zeros(eltype(θ), d, d)
    
    L = _chol_with_jitter(sampler.Σ_prop; jitter=sampler.jitter)
    
    m_init = model.model_builder(θ)
    _, loglik_curr = GeneralisedFilters.filter(rng, m_init, sampler.filter_algo, observations)
    
    n_accepted = 0
    @showprogress for i in 1:(n_samples + n_burnin)
        # Propose
        θ_prop = θ .+ (L * randn(rng, d))
        
        m_prop = model.model_builder(θ_prop)
        _, loglik_prop = GeneralisedFilters.filter(rng, m_prop, sampler.filter_algo, observations)
        
        log_alpha = (loglik_prop + logprior(θ_prop)) - (loglik_curr + logprior(θ))
        
        if log(rand(rng)) < log_alpha
            θ = θ_prop
            loglik_curr = loglik_prop
            n_accepted += 1
        end
        
        # Adaptation
        if sampler.adapt
            n_stats += 1
            if n_stats == 1
                μ_acc .= θ
            else
                δ = θ .- μ_acc
                μ_acc .+= δ ./ n_stats
                C_acc .+= δ * (θ .- μ_acc)'
            end
            
            if (i >= sampler.adapt_start) && (i <= sampler.adapt_end) && (i % sampler.adapt_interval == 0) && (n_stats > 1)
                Σ_emp = C_acc ./ (n_stats - 1)
                Σ_prop = sampler.scale * Matrix(Symmetric(Σ_emp)) + sampler.jitter * I(d)
                L = _chol_with_jitter(Σ_prop; jitter=sampler.jitter)
            end
        end
        
        if i > n_burnin
            samples[i - n_burnin] = deepcopy(θ)
        end
    end
    
    println("Acceptance rate: ", n_accepted / (n_samples + n_burnin))
    return samples
end

## PGibbs ######################################################################

struct PGibbs{F<:GeneralisedFilters.AbstractFilter, S<:Function} <: AbstractMCMC.AbstractSampler
    filter_algo::F
    θ_sampler::S
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::ParameterisedSSM,
    sampler::PGibbs,
    observations::AbstractVector;
    n_samples::Int,
    n_burnin::Int = 0,
    init_θ = nothing,
    ref_traj = nothing,
    kwargs...
)
    θ = init_θ === nothing ? rand(rng, model.prior) : init_θ
    samples = Vector{typeof(θ)}(undef, n_samples)
    ref_state = ref_traj
    
    @showprogress for i in 1:(n_samples + n_burnin)
        m = model.model_builder(θ)
        
        cb = GeneralisedFilters.DenseAncestorCallback(nothing)
        bf_state, _ = GeneralisedFilters.filter(rng, m, sampler.filter_algo, observations; ref_state=ref_state, callback=cb)
        
        w = softmax(getfield.(bf_state.particles, :log_w))
        idx = sample(1:length(w), Weights(w))
        ref_state = GeneralisedFilters.get_ancestry(cb.container, idx)
        
        θ = sampler.θ_sampler(ref_state, rng, θ)
        
        if i > n_burnin
            samples[i - n_burnin] = deepcopy(θ)
        end
    end
    
    return samples
end




## EHMM ########################################################################

struct EHMM{L, F<:GeneralisedFilters.AbstractFilter, S<:Function} <: AbstractMCMC.AbstractSampler
    filter_algo::F
    θ_sampler::S
end

function EHMM(filter_algo, θ_sampler, L::Int)
    return EHMM{L, typeof(filter_algo), typeof(θ_sampler)}(filter_algo, θ_sampler)
end

function EHMM(filter_algo, θ_sampler; L=15)
    return EHMM(filter_algo, θ_sampler, L)
end


function log_transition_density(model::StateSpaceModel, t, x_next, x_curr)
    # Generic transition density
    # Try to use SSMProblems interface first if available
    # For now, we rely on checking LinearGaussian properties or falling back
    
    dyn = SSMProblems.dyn(model)
    # Check if dyn has A, b, Q fields (Linear Gaussian)
    if hasproperty(dyn, :A) && hasproperty(dyn, :b) && hasproperty(dyn, :Q)
        # We assume time-invariant or we would need to access A[t] etc if they are arrays
        # But usually dyn.A is the matrix itself.
        # If the model is time-varying, A might be a function or vector.
        # For this research code, we assume HomogeneousLinearGaussianLatentDynamics
        dist = MvNormal(dyn.A * x_curr + dyn.b, dyn.Q)
        return logpdf(dist, x_next)
    else
         error("Generic transition density not implemented for $(typeof(dyn)). Only HomogeneousLinearGaussianLatentDynamics is supported in this research implementation.")
    end
end

function log_observation_density(model::StateSpaceModel, t, x, y)
    obs = SSMProblems.obs(model)
    if hasproperty(obs, :H) && hasproperty(obs, :c) && hasproperty(obs, :R)
        dist = MvNormal(obs.H * x + obs.c, obs.R)
        return logpdf(dist, y)
    else
         error("Generic observation density not implemented for $(typeof(obs)). Only HomogeneousLinearGaussianObservationProcess is supported.")
    end
end

function backward_simulation(rng::AbstractRNG, model::StateSpaceModel, particles, log_weights_T, observations)
    T = length(particles) - 1 # particles is 0:T
    
    # Pre-allocate trajectory
    traj = Vector{eltype(particles[T])}(undef, T + 1)
    
    # 1. Select x_T using the final weights from the filter
    ps_T = particles[T]
    w_T = softmax(log_weights_T)
    idx = sample(rng, 1:length(ps_T), Weights(w_T))
    traj[T + 1] = ps_T[idx]
    
    # 2. Backward pass
    for t in (T-1):-1:0
        x_next = traj[t+2] # x_{t+1}
        # y_curr = observations[t + 1] 
        
        ps = particles[t]
        log_ws = Vector{Float64}(undef, length(ps))
        
        for k in eachindex(ps)
            # Recompute weight: w_t \propto p(y_t | x_t).
            # If t=0, weight is 1 (log_w = 0).
            log_w_particle = (t == 0) ? 0.0 : log_observation_density(model, t, ps[k], observations[t])
            
            # BS weight: w_t * p(x_{t+1}|x_t)
            log_ws[k] = log_w_particle + log_transition_density(model, t, x_next, ps[k])
        end
        
        # Normalize weights
        max_log_w = maximum(log_ws)
        if max_log_w == -Inf
            ws = fill(1.0 / length(ps), length(ps))
        else
            ws = exp.(log_ws .- max_log_w)
            ws ./= sum(ws)
        end
        
        # Sample
        idx = sample(rng, 1:length(ps), Weights(ws))
        traj[t + 1] = ps[idx]
    end
    
    return OffsetArray(traj, 0:T)
end

function embedded_hmm_sampling(rng::AbstractRNG, model::StateSpaceModel, particles, ref_traj, L::Int, observations)
    _embedded_hmm_sampling(rng, model, particles, ref_traj, Val(L), observations)
end

function embedded_hmm_sampling(rng::AbstractRNG, model::StateSpaceModel, particles, ref_traj, ::Val{L}, observations) where {L}
    _embedded_hmm_sampling(rng, model, particles, ref_traj, Val(L), observations)
end

function _embedded_hmm_sampling(rng::AbstractRNG, model::StateSpaceModel, particles, ref_traj, ::Val{L}, observations) where {L}
    T = length(particles) - 1 # particles is 0:T
    
    # 1. Create Pools (SVector for pool, standard Vector for text)
    StateType = eltype(particles[0])
    PoolType = SVector{L, StateType}
    pools = Vector{PoolType}(undef, T + 1)
    
    for t in 0:T
        ps = particles[t]
        current_ref = ref_traj[t]
        
        # We sample indices
        inds = rand(rng, 1:length(ps), L - 1)
        
        # Construct pool using SVector with ntuple to handle non-isbits StateType
        pool_t = SVector{L, StateType}(ntuple(k -> k < L ? ps[inds[k]] : current_ref, Val(L)))
        pools[t + 1] = pool_t
    end
    
    # 2. Forward Pass
    # log_alpha[t][k] = P(x_t = pool[t][k] | y_{1:t})
    
    # Initialization (t=0)
    # alpha[0] is uniform 1/L.
    current_log_alpha = SVector{L, Float64}(fill(-log(L), L))
    
    # Store history for backward sampling
    all_log_alphas = Vector{SVector{L, Float64}}(undef, T + 1)
    all_log_alphas[1] = current_log_alpha
    
    for t in 1:T
        pool_prev = pools[t]     # pool at t-1
        pool_curr = pools[t + 1] # pool at t
        
        # Precompute observations density for current pool
        # SVector comprehension
        log_obs = SVector{L, Float64}([log_observation_density(model, t, pool_curr[j], observations[t]) for j in 1:L])
        
        # We compute next alphas
        # log_alpha_next[j] = log_obs[j] + logsumexp(log_alpha_prev[i] + log_trans[i,j])
        
        next_log_alpha_m = MVector{L, Float64}(undef)
        
        @inbounds for j in 1:L
            x_curr = pool_curr[j]
            
            # Compute transition log-probs from all i in prev
            log_trans_terms_m = MVector{L, Float64}(undef)
            for i in 1:L
                x_prev = pool_prev[i]
                log_trans_terms_m[i] = current_log_alpha[i] + log_transition_density(model, t - 1, x_curr, x_prev)
            end
            
            log_sum_trans = logsumexp(log_trans_terms_m)
            next_log_alpha_m[j] = log_obs[j] + log_sum_trans
        end
        
        current_log_alpha = SVector(next_log_alpha_m)
        all_log_alphas[t + 1] = current_log_alpha
    end
    
    # 3. Backward Sampling
    traj = Vector{StateType}(undef, T + 1)
    
    # Sample index at T
    log_alpha_T = all_log_alphas[T + 1]
    max_val = maximum(log_alpha_T)
    if max_val == -Inf
        w_T = SVector{L, Float64}(ntuple(_ -> 1.0/L, Val(L)))
    else
        w_T = softmax(log_alpha_T)
    end
    idx_T = sample(rng, 1:L, Weights(w_T))
    traj[T + 1] = pools[T + 1][idx_T]
    
    curr_idx = idx_T
    
    for t in (T-1):-1:0
        pool_curr = pools[t + 1] # pool at t
        pool_next = pools[t + 2] # pool at t+1
        
        x_next = pool_next[curr_idx]
        
        # Compute backward weights
        log_ws_m = MVector{L, Float64}(undef)
        @inbounds for i in 1:L
            x_curr = pool_curr[i]
            log_ws_m[i] = all_log_alphas[t + 1][i] + log_transition_density(model, t, x_next, x_curr)
        end
        
        log_ws_s = SVector(log_ws_m)
        max_val = maximum(log_ws_s)
        if max_val == -Inf
            w_t = SVector{L, Float64}(ntuple(_ -> 1.0/L, Val(L)))
        else
            w_t = softmax(log_ws_s)
        end
        curr_idx = sample(rng, 1:L, Weights(w_t))
        traj[t + 1] = pool_curr[curr_idx]
    end
    
    return OffsetArray(traj, 0:T)
end

function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::ParameterisedSSM,
    sampler::EHMM{L},
    observations::AbstractVector;
    n_samples::Int,
    n_burnin::Int = 0,
    init_θ = nothing,
    kwargs...
) where {L}
    theta = init_θ === nothing ? rand(rng, model.prior) : init_θ
    samples = Vector{typeof(theta)}(undef, n_samples)
    
    # Initialization: Run a standard Particle Filter to get a valid initial trajectory
    m_init = model.model_builder(theta)
    
    # Run filter WITH callback to get history
    cb_init = GeneralisedFilters.DenseAncestorCallback(nothing)
    bf_state, loglik_curr = GeneralisedFilters.filter(
        rng, 
        m_init, 
        sampler.filter_algo, 
        observations;
        callback = cb_init
    )
    
    # Initial backward simulation to get ref_traj (standard PGBS is fine for init)
    particles = cb_init.container.particles
    final_log_weights = getfield.(bf_state.particles, :log_w)
    ref_traj = backward_simulation(rng, m_init, particles, final_log_weights, observations)
    
    @showprogress for i in 1:(n_samples + n_burnin)
        m = model.model_builder(theta)
        
        # Run Conditional Particle Filter (CSMC)
        cb = GeneralisedFilters.DenseAncestorCallback(nothing)
        
        bf_state, loglik = GeneralisedFilters.filter(
            rng, 
            m, 
            sampler.filter_algo, 
            observations; 
            ref_state = ref_traj, 
            callback = cb
        )
        
        # Embedded HMM Sampling (Pool size L)
        particles = cb.container.particles
        ref_traj = embedded_hmm_sampling(rng, m, particles, ref_traj, Val(L), observations)
        
        # Sample Parameters given the trajectory
        theta = sampler.θ_sampler(ref_traj, rng, theta)
        
        if i > n_burnin
            samples[i - n_burnin] = deepcopy(theta)
        end
    end
    
    return samples
end

end
