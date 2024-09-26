using AnalyticalFilters
using Distributions
using LinearAlgebra
using Plots
using Random
using SSMProblems

import AnalyticalFilters: FilteringAlgorithm

struct TargetDynamics{T<:Real} <: LinearGaussianLatentDynamics{T}
    μ0::Vector{T}
    Σ0::Matrix{T}
    A::Matrix{T}
    b::Vector{T}
    Q::Matrix{T}
end

AnalyticalFilters.calc_μ0(dyn::TargetDynamics, extra) = dyn.μ0
AnalyticalFilters.calc_Σ0(dyn::TargetDynamics, extra) = dyn.Σ0
AnalyticalFilters.calc_A(dyn::TargetDynamics, ::Integer, extra) = dyn.A
AnalyticalFilters.calc_b(dyn::TargetDynamics, ::Integer, extra) = dyn.b
AnalyticalFilters.calc_Q(dyn::TargetDynamics, ::Integer, extra) = dyn.Q

struct TargetObservations{T<:Real} <: LinearGaussianObservationProcess{T}
    H::Matrix{T}
    R::Matrix{T}
end

AnalyticalFilters.calc_H(obs::TargetObservations, ::Integer, extra) = obs.H
function AnalyticalFilters.calc_c(obs::TargetObservations{T}, ::Integer, extra) where {T}
    return zeros(T, size(obs.H, 1))
end
AnalyticalFilters.calc_R(obs::TargetObservations, ::Integer, extra) = obs.R

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
struct ClutterAndTargetObservations{T} <: ObservationProcess{Vector{Vector{T}}}
    target_obs::TargetObservations{T}
    clutter_obs::ClutterObservations{T}
end

function SSMProblems.simulate(
    rng::AbstractRNG, obs::ClutterAndTargetObservations{T}, t::Integer, x::Vector, extra
) where {T}
    target_obs, clutter_obs = obs.target_obs, obs.clutter_obs
    target = SSMProblems.simulate(rng, target_obs, t, x, extra)
    clutter = SSMProblems.simulate(rng, clutter_obs, t, nothing, extra)
    return [[target]; clutter]
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

dyn = TargetDynamics(μ0, Σ0, A, b, Q)
target_obs = TargetObservations(H, R)
clutter_obs = ClutterObservations(λ, v)
obs = ClutterAndTargetObservations(target_obs, clutter_obs)

model = StateSpaceModel(dyn, obs)

rng = MersenneTwister(SEED)
x0, xs, ys = sample(rng, model, T)

true_x = getindex.(xs, 1)
true_y = getindex.(xs, 3)
plot(true_x, true_y; label="True trajectory", legend=:topleft)

t_test = T
meas_x = getindex.(ys[t_test], 1)
meas_y = getindex.(ys[t_test], 2)
scatter!(meas_x, meas_y; label="Measurements")

###################
#### FILTERING ####
###################

abstract type Associator end

struct NearestNeighborAssociator <: Associator end

# x_t
# -> x_{t+1}_pred
# -> y_{t+1}_pred = H x_{t+1}_pred (noiseless)
# -> y_{t+1} compare with y_{t+1}_pred

function associate(::NearestNeighborAssociator, x::Vector, ys::Vector)
    return argmin(norm.([x] .- ys))
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
    model::StateSpaceModel{<:Any,<:ClutterAndTargetObservations},
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

function step(
    model::StateSpaceModel{<:Any,<:ClutterAndTargetObservations},
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
    y_pred = model.obs.target_obs.H * state.μ
    # y_pred = simulate(model.obs, t, state.μ, extra, noise=false)  # H * x or H * x + R
    assoc = associate(assoc_filter.associator, y_pred, ys)
    state, ll = AnalyticalFilters.update(
        conditioned_model, assoc_filter.filter, t, state, ys[assoc], extra
    )
    return state, ll
end

# Test filter runs
assoc_filter = AssociationFilter(NearestNeighborAssociator(), KalmanFilter())
extras = fill(nothing, T)
state, ll = filter(model, assoc_filter, ys, nothing, extras)
