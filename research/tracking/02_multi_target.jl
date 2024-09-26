using AnalyticalFilters
using Distributions
using LinearAlgebra
using Plots
using Random
using SSMProblems

import AnalyticalFilters: FilteringAlgorithm

struct MultiTargetDynamics{T<:Real} <: LinearGaussianLatentDynamics{T}
    μ0s::Vector{Vector{T}}
    Σ0s::Vector{Matrix{T}}
    As::Vector{Matrix{T}}
    bs::Vector{Vector{T}}
    Qs::Vector{Matrix{T}}
    N::Int
    D::Int
end

AnalyticalFilters.calc_μ0(dyn::MultiTargetDynamics, extra) = vcat(dyn.μ0s...)
function AnalyticalFilters.calc_Σ0(dyn::MultiTargetDynamics, extra)
    Σ0 = zeros(dyn.D * dyn.N, dyn.D * dyn.N)
    for i in 1:(dyn.N)
        idxs = ((i - 1) * dyn.D + 1):(i * dyn.D)
        Σ0[idxs, idxs] .= dyn.Σ0s[i]
    end
    return Σ0
end
function AnalyticalFilters.calc_A(dyn::MultiTargetDynamics, ::Integer, extra)
    A = zeros(dyn.D * dyn.N, dyn.D * dyn.N)
    for i in 1:(dyn.N)
        idxs = ((i - 1) * dyn.D + 1):(i * dyn.D)
        A[idxs, idxs] .= dyn.As[i]
    end
    return A
end
AnalyticalFilters.calc_b(dyn::MultiTargetDynamics, ::Integer, extra) = vcat(dyn.bs...)
function AnalyticalFilters.calc_Q(dyn::MultiTargetDynamics, ::Integer, extra)
    Q = zeros(dyn.D * dyn.N, dyn.D * dyn.N)
    for i in 1:(dyn.N)
        idxs = ((i - 1) * dyn.D + 1):(i * dyn.D)
        Q[idxs, idxs] .= dyn.Qs[i]
    end
    return Q
end

struct MultiTargetObservations{T<:Real} <: LinearGaussianObservationProcess{T}
    Hs::Vector{Matrix{T}}
    Rs::Vector{Matrix{T}}
    N::Int
    D_x::Int
    D_y::Int
end

function AnalyticalFilters.calc_H(obs::MultiTargetObservations, ::Integer, extra)
    H = zeros(obs.D_y * obs.N, obs.D_x * obs.N)
    for i in 1:(obs.N)
        idxs_x = ((i - 1) * obs.D_x + 1):(i * obs.D_x)
        idxs_y = ((i - 1) * obs.D_y + 1):(i * obs.D_y)
        H[idxs_y, idxs_x] .= obs.Hs[i]
    end
    return H
end
function AnalyticalFilters.calc_c(
    obs::MultiTargetObservations{T}, ::Integer, extra
) where {T}
    return zeros(T, obs.D_y * obs.N)
end
function AnalyticalFilters.calc_R(obs::MultiTargetObservations, ::Integer, extra)
    R = zeros(obs.D_y * obs.N, obs.D_y * obs.N)
    for i in 1:(obs.N)
        idxs = ((i - 1) * obs.D_y + 1):(i * obs.D_y)
        R[idxs, idxs] .= obs.Rs[i]
    end
    return R
end

struct ClutterObservations{T<:Real} <: ObservationProcess{Vector{Vector{T}}}
    λ::T
    v::T
end

function SSMProblems.simulate(
    rng::AbstractRNG, obs::ClutterObservations{T}, ::Integer, ::Nothing, extra
) where {T}
    n = rand(rng, Poisson(obs.λ))
    return [obs.v * (T(2) * rand(rng, T, 2) .- T(1)) for _ in 1:n]
end

# TODO: add detection rate
struct ClutterAndMultiTargetObservations{T} <: ObservationProcess{Vector{Vector{T}}}
    target_obs::MultiTargetObservations{T}
    clutter_obs::ClutterObservations{T}
end

function SSMProblems.simulate(
    rng::AbstractRNG,
    obs::ClutterAndMultiTargetObservations{T},
    t::Integer,
    x::Vector,
    extra,
) where {T}
    target_obs, clutter_obs = obs.target_obs, obs.clutter_obs
    targets = SSMProblems.simulate(rng, target_obs, t, x, extra)
    clutter = SSMProblems.simulate(rng, clutter_obs, t, nothing, extra)
    D, N = target_obs.D_y, target_obs.N
    # TODO: get rid of multi target dynamics and instead call Kalman filter iteratively
    targets = [targets[(D * (i - 1) + 1):(D * i)] for i in 1:N]
    return [targets; clutter]
end

T = 100;
SEED = 1234;

μ0 = [-1.0, 0.0, -2.0, 0.0];
Σ0 = Matrix(1.0I, 4, 4);
A = [1.0 1.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 1.0; 0.0 0.0 0.0 1.0];
b = zeros(4);
Q = [
    1/3 1/2 0 0
    1/2 1 0 0
    0 0 1/3 1/2
    0 0 1/2 1
];
H = [1.0 0.0 0.0 0.0; 0.0 0.0 1.0 0.0];
R = Matrix(0.3I, 2, 2);
v = 500.0
# pd = 0.9
λ = 50.0

N = 4
μ0s = repeat([μ0], N)
Σ0s = repeat([Σ0], N)
As = repeat([A], N)
bs = repeat([b], N)
Qs = repeat([Q], N)
Hs = repeat([H], N)
Rs = repeat([R], N)

dyn = MultiTargetDynamics(μ0s, Σ0s, As, bs, Qs, N, 4)
target_obs = MultiTargetObservations(Hs, Rs, N, 4, 2)
clutter_obs = ClutterObservations(λ, v)
obs = ClutterAndMultiTargetObservations(target_obs, clutter_obs)

model = StateSpaceModel(dyn, obs)

rng = MersenneTwister(SEED)
x0, xs, ys = sample(rng, model, T)

true_x = getindex.(xs, 1)
true_y = getindex.(xs, 3)
plot(true_x, true_y; label="True trajectory", legend=:topleft)

true_x = getindex.(xs, 5)
true_y = getindex.(xs, 7)
plot!(true_x, true_y; label="True trajectory", legend=:topleft)

t_test = T
meas_x = getindex.(ys[t_test], 1)
meas_y = getindex.(ys[t_test], 2)
scatter!(meas_x, meas_y; label="Measurements")

###################
#### FILTERING ####
###################

abstract type Associator end

struct GlobalNearestNeighborAssociator <: Associator end

# x_t
# -> x_{t+1}_pred
# -> y_{t+1}_pred = H x_{t+1}_pred (noiseless)
# -> y_{t+1} compare with y_{t+1}_pred

function associate(::GlobalNearestNeighborAssociator, y_preds::Vector, ys::Vector)
    # HACK: very inefficient
    ys_copy = copy(ys)
    assocs = Vector{Int}(undef, length(y_preds))
    for (i, y_pred) in enumerate(y_preds)
        idx = argmin(norm.([y_pred] .- ys_copy))
        assocs[i] = idx
        deleteat!(ys_copy, idx)
    end
    return assocs
end

struct AssociationFilter{A<:Associator,F<:FilteringAlgorithm} <: FilteringAlgorithm
    associator::A
    filter::F
end

# HACK: this should be combined with usual initialisation function which should only use
# dynamics
function initialise(model, filter, extra)
    μ0, Σ0 = AnalyticalFilters.calc_initial(model.dyn, extra)
    return (μ=μ0, Σ=Σ0)
end

function filter(
    model::StateSpaceModel{<:Any,<:ClutterAndMultiTargetObservations},
    assoc_filter::AssociationFilter,
    Ys::Vector{Vector{Vector{Float64}}},
    extra0,
    extras,
)
    state = initialise(model, assoc_filter.filter, extra0)
    ll = 0.0
    for (t, ys) in enumerate(Ys)
        state, l = step(model, assoc_filter, t, state, ys, extras[t])
        ll += l
    end
    return state, ll
end

history = Vector{Vector{Float64}}(undef, T)

function step(
    model::StateSpaceModel{<:Any,<:ClutterAndMultiTargetObservations},
    assoc_filter::AssociationFilter,
    t::Integer,
    state,
    ys::Vector{Vector{Float64}},
    extra,
)
    # HACK: need a clean way to pass conditioned model to the analytical filter
    conditioned_model = StateSpaceModel(model.dyn, model.obs.target_obs)

    state = AnalyticalFilters.predict(
        conditioned_model, assoc_filter.filter, t, state, extra
    )
    # HACK: assuming form of state
    N, D = model.obs.target_obs.N, model.obs.target_obs.D_x
    μ_vec_vec = [state.μ[(D * (i - 1) + 1):(D * i)] for i in 1:N]
    y_preds = model.obs.target_obs.Hs .* μ_vec_vec
    # y_pred = simulate(model.obs, t, state.μ, extra, noise=false)  # H * x or H * x + R
    assocs = associate(assoc_filter.associator, y_preds, ys)
    state, ll = AnalyticalFilters.update(
        conditioned_model, assoc_filter.filter, t, state, vcat(ys[assocs]...), extra
    )

    history[t] = copy(state.μ)

    return state, ll
end

# Test filter runs
assoc_filter = AssociationFilter(GlobalNearestNeighborAssociator(), KalmanFilter())
extras = fill(nothing, T)
state, ll = filter(model, assoc_filter, ys, nothing, extras)

filt_x = getindex.(history, 1)
filt_y = getindex.(history, 3)
plot!(filt_x, filt_y; label="Filtered trajectory", legend=:topleft)
