"""
Helpers for testing analytical gradients through the normalized discrete/HMM filter.
"""

function _softmax(x::AbstractVector)
    shifted = x .- maximum(x)
    ex = exp.(shifted)
    return ex / sum(ex)
end

function _row_softmax(Ψ::AbstractMatrix)
    P = similar(Ψ)
    for i in axes(Ψ, 1)
        row = vec(Ψ[i, :])
        P[i, :] .= _softmax(row)
    end
    return P
end

function _softmax_pullback(∂p::AbstractVector, p::AbstractVector)
    return p .* (∂p .- dot(∂p, p))
end

"""
    setup_discrete_gradient_test(rng; K=3, T=4)

Set up a finite-state HMM scenario and compute analytical NLL gradients through the
normalized filtering recursion.
"""
function setup_discrete_gradient_test(rng::AbstractRNG; K::Int=3, T::Int=4)
    α0 = rand(rng, K)
    α0 ./= sum(α0)

    P = rand(rng, K, K) .+ 0.1
    P ./= sum(P; dims=2)

    μs = collect(range(-1.0; step=1.0, length=K))

    prior = HomogeneousDiscretePrior(α0)
    dyn = HomogeneousDiscreteLatentDynamics(P)
    obs = MixtureObservation(μs)
    model = StateSpaceModel(prior, dyn, obs)

    _, _, ys = SSMProblems.sample(rng, model, T)

    predict_caches = Vector{GeneralisedFilters.DiscretePredictGradientCache}(undef, T)
    update_caches = Vector{GeneralisedFilters.DiscreteGradientCache}(undef, T)

    state = GeneralisedFilters.initialise(rng, prior, GeneralisedFilters.DF())
    for t in 1:T
        pred, predict_caches[t] = GeneralisedFilters.predict_with_cache(
            rng, dyn, GeneralisedFilters.DF(), t, state, ys[t]
        )
        state, _, update_caches[t] = GeneralisedFilters.update_with_cache(
            obs, GeneralisedFilters.DF(), t, pred, ys[t]
        )
    end

    ∂filtered = zeros(K)
    ∂P_total = zeros(K, K)
    ∂logb_by_t = [zeros(K) for _ in 1:T]

    for t in T:-1:1
        ∂pred, ∂b = GeneralisedFilters.backward_gradient_update(∂filtered, update_caches[t])
        ∂P_total .+= GeneralisedFilters.gradient_P(∂pred, predict_caches[t])
        ∂logb_by_t[t] .= GeneralisedFilters.gradient_log_emission(∂b, update_caches[t])
        ∂filtered = GeneralisedFilters.backward_gradient_predict(∂pred, predict_caches[t])
    end

    ∂α0 = ∂filtered

    # For MixtureObservation: log b_t(i) = log N(y_t; μ_i, 1), so ∂ log b_t(i) / ∂μ_i = y_t - μ_i
    ∂μ_total = zeros(K)
    for t in 1:T
        y_t = ys[t]
        ∂μ_total .+= ∂logb_by_t[t] .* (y_t .- μs)
    end

    # Map simplex gradients to unconstrained logits for robust FD comparison.
    η0 = log.(α0)
    Ψ = log.(P)

    ∂η0 = _softmax_pullback(∂α0, α0)
    ∂Ψ_total = zeros(K, K)
    for i in 1:K
        ∂Ψ_total[i, :] .= _softmax_pullback(vec(∂P_total[i, :]), vec(P[i, :]))
    end

    return (;
        K,
        T,
        model,
        ys,
        α0,
        P,
        μs,
        η0,
        Ψ,
        ∂α0,
        ∂P_total,
        ∂μ_total,
        ∂η0,
        ∂Ψ_total,
    )
end

"""
    make_discrete_nll_func(model, ys, param::Symbol)

Create a finite-difference-ready NLL function for discrete gradient tests.
Supported symbols: `:η0`, `:Ψ`, `:μ`.
"""
function make_discrete_nll_func(model, ys, param::Symbol)
    prior = SSMProblems.prior(model)
    dyn = SSMProblems.dyn(model)
    obs = SSMProblems.obs(model)

    α0 = GeneralisedFilters.calc_α0(prior)
    P = GeneralisedFilters.calc_P(dyn, 1)
    μs = obs.μs
    K = length(α0)

    if param == :η0
        return function (η)
            α0_new = _softmax(η)
            m = StateSpaceModel(
                HomogeneousDiscretePrior(α0_new), HomogeneousDiscreteLatentDynamics(P), MixtureObservation(μs)
            )
            _, ll = GeneralisedFilters.filter(m, GeneralisedFilters.DF(), ys)
            return -ll
        end
    elseif param == :Ψ
        return function (Ψ_vec)
            Ψ_new = reshape(Ψ_vec, K, K)
            P_new = _row_softmax(Ψ_new)
            m = StateSpaceModel(
                HomogeneousDiscretePrior(α0), HomogeneousDiscreteLatentDynamics(P_new), MixtureObservation(μs)
            )
            _, ll = GeneralisedFilters.filter(m, GeneralisedFilters.DF(), ys)
            return -ll
        end
    elseif param == :μ
        return function (μ)
            m = StateSpaceModel(
                HomogeneousDiscretePrior(α0), HomogeneousDiscreteLatentDynamics(P), MixtureObservation(μ)
            )
            _, ll = GeneralisedFilters.filter(m, GeneralisedFilters.DF(), ys)
            return -ll
        end
    else
        error("Unknown parameter: $param")
    end
end
