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
    return _kalman_state(prior.μ0, prior.Σ0)
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
Observations must be nonempty. The likelihood total starts with the first increment and
retains the scalar type determined by the Kalman calculations.
"""
function marginal_loglikelihood(
    model::StateSpaceModel, af::KalmanFilter, ys::AbstractVector
)
    _validate_observations(model, ys)
    p = model.prior::GaussianPrior
    isempty(ys) &&
        throw(ArgumentError("Kalman marginal_loglikelihood requires nonempty observations"))
    state = _kalman_state(p.μ0, p.Σ0)
    state, ll = _kalman_likelihood_step(model, af, state, 1, ys[1])
    # The first observation/transition may promote the initial state's scalar type.
    # Start a separately specialised loop with that promoted state and increment.
    return _kalman_likelihood_tail(model, af, ys, state, ll)
end

function _kalman_likelihood_step(model, af, state, t, y)
    d = resolve(model.dyn, (; t))
    o = resolve(model.obs, (; t))
    state, inc = kalman_step(state, d, o, y)
    return _kalman_state(state.μ, repair_covariance(af.repair, state.Σ)), inc
end

Base.@noinline function _kalman_likelihood_tail(model, af, ys, state, ll)
    for t in 2:length(ys)
        state, inc = _kalman_likelihood_step(model, af, state, t, ys[t])
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
    pred = predict(rng, model.dyn, kf, 1, state, ys[1])
    state, total_ll = update(model.obs, kf, 1, pred, ys[1])
    # The first update may promote both precision and storage (e.g. a Float32
    # prior with Float64 observations). Store its prediction in the promoted
    # representation too, without changing its values or the model parameters.
    first_pred = GaussianState(typeof(state.μ)(pred.μ), typeof(state.Σ)(pred.Σ))
    predicted = Vector{typeof(first_pred)}(undef, T)
    filtered = Vector{typeof(state)}(undef, T)
    predicted[1], filtered[1] = first_pred, state

    for t in 2:T
        pred = predict(rng, model.dyn, kf, t, state, ys[t])
        state, ll = update(model.obs, kf, t, pred, ys[t])
        (typeof(pred) === eltype(predicted) && typeof(state) === eltype(filtered)) || throw(
            ArgumentError(
                "Kalman smoothing requires stable state storage and scalar types after the first update",
            ),
        )
        predicted[t], filtered[t] = pred, state
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
