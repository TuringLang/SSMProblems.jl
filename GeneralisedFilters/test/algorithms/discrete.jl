"""Unit tests for discrete state (HMM) algorithms."""

## Forward Filtering ########################################################################

@testitem "Discrete filter" begin
    using GeneralisedFilters
    using GeneralisedFilters.GFTest: MixtureObservation
    using Distributions
    using StableRNGs
    using SSMProblems

    rng = StableRNG(1234)
    α0 = rand(rng, 3)
    α0 = α0 / sum(α0)
    P = rand(rng, 3, 3)
    P = P ./ sum(P; dims=2)

    μs = [0.0, 1.0, 2.0]

    prior = HomogeneousDiscretePrior(α0)
    dyn = HomogeneousDiscreteLatentDynamics(P)
    obs = MixtureObservation(μs)
    model = StateSpaceModel(prior, dyn, obs)

    observations = [rand(rng)]

    df = DiscreteFilter()
    state, ll = GeneralisedFilters.filter(model, df, observations)

    # Brute force calculations of each conditional path probability p(x_{1:T} | y_{1:T})
    T = 1
    K = 3
    y = only(observations)
    path_probs = Dict{Tuple{Int,Int},Float64}()
    for x0 in 1:K, x1 in 1:K
        prior_prob = α0[x0] * P[x0, x1]
        likelihood = exp(SSMProblems.logdensity(obs, 1, x1, y))
        path_probs[(x0, x1)] = prior_prob * likelihood
    end
    marginal = sum(values(path_probs))

    filtered_paths = Base.filter(((k, v),) -> k[end] == 1, path_probs)
    @test state[1] ≈ sum(values(filtered_paths)) / marginal
    @test ll ≈ log(marginal)
end

## Backward Filtering #######################################################################

@testitem "Backward discrete predictor" begin
    using GeneralisedFilters
    using GeneralisedFilters.GFTest: MixtureObservation
    using Distributions
    using Random
    using SSMProblems
    using LogExpFunctions

    # Simple 3-state HMM with Gaussian emissions
    K = 3
    T = 4

    α0 = [0.5, 0.3, 0.2]
    P = [
        0.7 0.2 0.1
        0.1 0.8 0.1
        0.2 0.2 0.6
    ]

    μs = [0.0, 2.0, 4.0]
    obs = MixtureObservation(μs)

    observations = [0.5, 1.8, 3.5, 2.1]

    # Run backward predictor
    algo = BackwardDiscretePredictor()
    rng = Random.default_rng()
    dyn = HomogeneousDiscreteLatentDynamics(P)

    # Initialize at time T and run backward pass
    β = (
        let lik = GeneralisedFilters.backward_initialise(
                rng, obs, algo, T, observations[T]; num_states=K
            )
            for t in (T - 1):-1:1
                lik = GeneralisedFilters.backward_predict(rng, dyn, algo, t, lik)
                lik = GeneralisedFilters.backward_update(obs, algo, t, lik, observations[t])
            end
            lik
        end
    )

    # Brute force: compute β_1(i) = p(y_{1:T} | x_1 = i) by enumerating all paths
    log_β_bruteforce = zeros(K)
    for x1 in 1:K
        log_prob = -Inf
        for x2 in 1:K, x3 in 1:K, x4 in 1:K
            log_path_prob = 0.0
            # Transitions
            log_path_prob += log(P[x1, x2]) + log(P[x2, x3]) + log(P[x3, x4])
            # Emissions
            log_path_prob += logpdf(Normal(μs[x1], 1.0), observations[1])
            log_path_prob += logpdf(Normal(μs[x2], 1.0), observations[2])
            log_path_prob += logpdf(Normal(μs[x3], 1.0), observations[3])
            log_path_prob += logpdf(Normal(μs[x4], 1.0), observations[4])
            log_prob = logaddexp(log_prob, log_path_prob)
        end
        log_β_bruteforce[x1] = log_prob
    end

    @test log_likelihoods(β) ≈ log_β_bruteforce
end

## RTS-style Smoothing ######################################################################

@testitem "Discrete smoother" begin
    using GeneralisedFilters
    using GeneralisedFilters.GFTest: MixtureObservation
    using Distributions
    using Random
    using SSMProblems
    using LogExpFunctions

    K = 3
    T = 4
    t_smooth = 2

    α0 = [0.5, 0.3, 0.2]
    P = [
        0.7 0.2 0.1
        0.1 0.8 0.1
        0.2 0.2 0.6
    ]

    μs = [0.0, 2.0, 4.0]
    obs = MixtureObservation(μs)

    prior = HomogeneousDiscretePrior(α0)
    dyn = HomogeneousDiscreteLatentDynamics(P)
    model = StateSpaceModel(prior, dyn, obs)

    observations = [0.5, 1.8, 3.5, 2.1]

    rng = Random.default_rng()
    smoothed, ll = smooth(rng, model, DiscreteSmoother(), observations; t_smooth=t_smooth)

    # Brute force: compute γ_{t_smooth}(i) = p(x_{t_smooth} = i | y_{1:T})
    # by enumerating all paths and marginalizing
    log_joint_probs = Dict{NTuple{5,Int},Float64}()
    for x0 in 1:K, x1 in 1:K, x2 in 1:K, x3 in 1:K, x4 in 1:K
        log_prob = 0.0
        log_prob += log(α0[x0])
        log_prob += log(P[x0, x1]) + log(P[x1, x2]) + log(P[x2, x3]) + log(P[x3, x4])
        log_prob += logpdf(Normal(μs[x1], 1.0), observations[1])
        log_prob += logpdf(Normal(μs[x2], 1.0), observations[2])
        log_prob += logpdf(Normal(μs[x3], 1.0), observations[3])
        log_prob += logpdf(Normal(μs[x4], 1.0), observations[4])
        log_joint_probs[(x0, x1, x2, x3, x4)] = log_prob
    end

    log_marginal = logsumexp(collect(values(log_joint_probs)))

    # Marginalize to get p(x_{t_smooth} | y_{1:T})
    # t_smooth=2 corresponds to x2 (index 3 in the tuple since x0 is index 1)
    γ_bruteforce = zeros(K)
    for i in 1:K
        matching_paths = Base.filter(((k, v),) -> k[t_smooth + 1] == i, log_joint_probs)
        if !isempty(matching_paths)
            γ_bruteforce[i] = exp(logsumexp(collect(values(matching_paths))) - log_marginal)
        end
    end

    @test smoothed ≈ γ_bruteforce
    @test ll ≈ log_marginal
end

## Two-Filter Smoothing #####################################################################

@testitem "Discrete two-filter smoother" begin
    using GeneralisedFilters
    using GeneralisedFilters.GFTest: MixtureObservation
    using Distributions
    using Random
    using SSMProblems
    using LogExpFunctions

    K = 3
    T = 4
    t_smooth = 2

    α0 = [0.5, 0.3, 0.2]
    P = [
        0.7 0.2 0.1
        0.1 0.8 0.1
        0.2 0.2 0.6
    ]

    μs = [0.0, 2.0, 4.0]
    obs = MixtureObservation(μs)

    prior = HomogeneousDiscretePrior(α0)
    dyn = HomogeneousDiscreteLatentDynamics(P)
    model = StateSpaceModel(prior, dyn, obs)

    observations = [0.5, 1.8, 3.5, 2.1]

    rng = Random.default_rng()

    # Forward pass to get filtered distribution at t_smooth
    df = DiscreteFilter()
    filtered = Vector{Vector{Float64}}(undef, T)

    let s = initialise(rng, SSMProblems.prior(model), df)
        for t in 1:T
            pred = predict(rng, SSMProblems.dyn(model), df, t, s, observations[t])
            s, _ = update(SSMProblems.obs(model), df, t, pred, observations[t])
            filtered[t] = s
        end
    end

    # Backward pass to get predictive likelihood at t_smooth
    # β_{t_smooth}(i) = p(y_{t_smooth+1:T} | x_{t_smooth} = i)
    bdp = BackwardDiscretePredictor()
    back_lik = (
        let lik = GeneralisedFilters.backward_initialise(
                rng, obs, bdp, T, observations[T]; num_states=K
            )
            for t in (T - 1):-1:(t_smooth + 1)
                lik = GeneralisedFilters.backward_predict(rng, dyn, bdp, t, lik)
                lik = GeneralisedFilters.backward_update(obs, bdp, t, lik, observations[t])
            end
            GeneralisedFilters.backward_predict(rng, dyn, bdp, t_smooth, lik)
        end
    )

    # Two-filter smooth
    smoothed_2f = two_filter_smooth(filtered[t_smooth], back_lik)

    # Compare to RTS smoother
    smoothed_rts, _ = smooth(
        rng, model, DiscreteSmoother(), observations; t_smooth=t_smooth
    )

    @test smoothed_2f ≈ smoothed_rts
end
