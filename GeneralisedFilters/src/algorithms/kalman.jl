export KalmanFilter, KF, KalmanSmoother, KS
export BackwardInformationPredictor
export marginal_loglikelihood

"""
    KalmanFilter(; repair=NoRepair())

Kalman filter for linear-Gaussian state-space models. `repair` is a [`CovarianceRepair`](@ref)
strategy applied to the filtered covariance after each update.
"""
struct KalmanFilter{RT<:CovarianceRepair} <: AbstractFilter
    repair::RT
end
KalmanFilter(; repair=NoRepair()) = KalmanFilter(repair)
KF() = KalmanFilter()

function initialise(::AbstractRNG, prior::GaussianPrior, ::KalmanFilter; ref_state=nothing)
    return GaussianState(prior.μ0, prior.Σ0)
end

function predict(
    ::AbstractRNG,
    dyn,
    algo::KalmanFilter,
    t::Integer,
    state::GaussianState,
    y;
    ref_state=nothing,
)
    return kalman_predict(state, _component(resolve(dyn, (; t))))
end

function update(obs, algo::KalmanFilter, t::Integer, state::GaussianState, y)
    return kalman_update(state, _component(resolve(obs, (; t))), y; repair=algo.repair)
end

"""
    marginal_loglikelihood(model, af::KalmanFilter, ys)

Marginal log-likelihood `log p(y_{1:T})` of a linear-Gaussian model, computed through the
fused Kalman step. Conditional inner models returned by `condition_inner` use this same
evaluator. Observations must be one-based and match the conditional trajectory horizon.
An empty observation sequence has zero marginal log-likelihood.
"""
function marginal_loglikelihood(
    model::StateSpaceModel, af::KalmanFilter, ys::AbstractVector
)
    _validate_observations(model, ys)
    p = model.prior::GaussianPrior
    state = GaussianState(p.μ0, p.Σ0)
    ll = zero(eltype(p.μ0))
    for t in eachindex(ys)
        d = resolve(model.dyn, (; t))
        o = resolve(model.obs, (; t))
        state, inc = kalman_step(state, d, o, ys[t])
        state = GaussianState(state.μ, repair_covariance(af.repair, state.Σ))
        ll += inc
    end
    return ll
end

## KALMAN SMOOTHER #########################################################################

struct KalmanSmoother <: AbstractSmoother end
const KS = KalmanSmoother()

function smooth(
    rng::AbstractRNG,
    model::StateSpaceModel,
    ::KalmanSmoother,
    ys::AbstractVector;
    t_smooth=1,
)
    _validate_observations(model, ys)
    1 <= t_smooth <= length(ys) ||
        throw(ArgumentError("smoothing time must lie in 1:length(ys)"))
    kf = KalmanFilter()
    T = length(ys)

    state = initialise(rng, model.prior, kf)
    GS = typeof(state)
    predicted = Vector{GS}(undef, T)
    filtered = Vector{GS}(undef, T)

    total_ll = zero(eltype(state))
    for t in 1:T
        pred = predict(rng, model.dyn, kf, t, state, ys[t])
        predicted[t] = pred
        state, ll = update(model.obs, kf, t, pred, ys[t])
        filtered[t] = state
        total_ll += ll
    end

    smoothed = filtered[T]
    for t in (T - 1):-1:t_smooth
        # Atom index t+1 parameterises the transition x_t → x_{t+1}.
        d = _component(resolve(model.dyn, (; t=t + 1)))
        smoothed = rts_backward_step(filtered[t], d, smoothed, predicted[t + 1])
    end

    return smoothed, total_ll
end
function smooth(model::StateSpaceModel, algo::KalmanSmoother, ys::AbstractVector; kwargs...)
    return smooth(default_rng(), model, algo, ys; kwargs...)
end
