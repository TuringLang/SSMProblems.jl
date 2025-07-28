using GeneralisedFilters
using SSMProblems
using Distributions

using Random
using StatsBase
using LinearAlgebra

include("utilities.jl")

const GF = GeneralisedFilters

## TREND DYNAMICS ##########################################################################

struct LocalLevelTrend <: LinearGaussianLatentDynamics end

GF.calc_A(::LocalLevelTrend, ::Integer; kwargs...) = [1;;]
GF.calc_b(::LocalLevelTrend, ::Integer; kwargs...) = [0;]

function GF.calc_Q(::LocalLevelTrend, ::Integer; new_outer, kwargs...)
    return [exp(new_outer[1]);;]
end

## OBSERVATION PROCESSES ###################################################################

struct SimpleObservation <: LinearGaussianObservationProcess end

GF.calc_H(::SimpleObservation, ::Integer; kwargs...) = [1;;]
GF.calc_c(::SimpleObservation, ::Integer; kwargs...) = [0;]

function GF.calc_R(::SimpleObservation, ::Integer; new_outer, kwargs...)
    return [exp(new_outer[2]);;]
end

struct OutlierAdjustedObservation <: LinearGaussianObservationProcess end

GF.calc_H(::OutlierAdjustedObservation, ::Integer; kwargs...) = [1;;]
GF.calc_c(::OutlierAdjustedObservation, ::Integer; kwargs...) = [0;]

function GF.calc_R(::OutlierAdjustedObservation, ::Integer; new_outer, kwargs...)
    return [new_outer[3] * exp(new_outer[2]);;]
end

## VOLATILITY PRIORS #######################################################################

struct StochasticVolatilityPrior{T<:Real} <: StatePrior end

function SSMProblems.distribution(
    prior::StochasticVolatilityPrior{T}; kwargs...
) where {T}
    return product_distribution(Normal(zero(T), T(1)), Normal(zero(T), T(1)))
end

struct OutlierAdjustedVolatilityPrior{T<:Real} <: StatePrior end

function SSMProblems.distribution(
    prior::OutlierAdjustedVolatilityPrior{T}; kwargs...
) where {T}
    return product_distribution(Normal(zero(T), T(1)), Normal(zero(T), T(1)), Dirac(one(T)))
end

## STOCHASTIC VOLATILITY PROCESS ###########################################################

struct StochasticVolatility{ΓT<:AbstractVector} <: LatentDynamics
    γ::ΓT
end

function SSMProblems.simulate(
    rng::AbstractRNG,
    proc::StochasticVolatility,
    step::Integer,
    state::AbstractVector{T};
    kwargs...
) where {T<:Real}
    new_state = deepcopy(state)
    new_state[1:2] += proc.γ .* randn(rng, T, 2)
    return new_state
end

struct OutlierAdjustedVolatility{ΓT} <: LatentDynamics
    volatility::StochasticVolatility{ΓT}
    switch_dist::Bernoulli
    outlier_dist::Uniform
end

function SSMProblems.simulate(
    rng::AbstractRNG,
    proc::OutlierAdjustedVolatility,
    step::Integer,
    state::AbstractVector{T};
    kwargs...,
) where {T<:Real}
    new_state = SSMProblems.simulate(rng, proc.volatility, step, state; kwargs...)
    new_state[3] = rand(rng, proc.switch_dist) ? rand(rng, proc.outlier_dist) : one(T)
    return new_state
end

## MAIN ####################################################################################

function UCSV(γ::T) where {T<:Real}
    # volatility dynamics
    stoch_vol_prior = StochasticVolatilityPrior{T}()
    stoch_vol_process = StochasticVolatility(fill(γ, 2))

    # conditionally linear and Gaussian trend model
    local_level_model = StateSpaceModel(
        GF.HomogeneousGaussianPrior(zeros(T, 1), Matrix(100.0I(1))),
        LocalLevelTrend(),
        SimpleObservation()
    )

    return HierarchicalSSM(
        stoch_vol_prior, stoch_vol_process, local_level_model
    )
end

function UCSVO(γ::T, prob::T) where {T<:Real}
    # volatility dynamics and outlier probabilities
    stoch_vol_prior = OutlierAdjustedVolatilityPrior{T}()
    stoch_vol_process = OutlierAdjustedVolatility(
        StochasticVolatility(fill(γ, 2)), Bernoulli(prob), Uniform{T}(2, 10)
    )

    # conditionally linear and Gaussian trend model
    local_level_model = StateSpaceModel(
        GF.HomogeneousGaussianPrior(zeros(T, 1), Matrix(100.0I(1))),
        LocalLevelTrend(),
        OutlierAdjustedObservation()
    )

    return HierarchicalSSM(
        stoch_vol_prior, stoch_vol_process, local_level_model
    )
end

function plot_ucsv(rng::AbstractRNG, model::HierarchicalSSM, data)
    alg = RBPF(KalmanFilter(), 2^12; threshold=1.0)
    sparse_ancestry = GF.AncestorCallback(nothing)
    states, ll = GF.filter(rng, model, alg, data; callback=sparse_ancestry)

    fig = Figure(; size=(1200, 500), fontsize=16)
    dateticks = date_format(fred.data.date)
    all_paths = GF.get_ancestry(sparse_ancestry.tree)

    zs = mean(
        [hcat(getproperty.(getproperty.(path, :z), :μ)...) for path in all_paths],
        Weights(StatsBase.weights(states.log_weights))
    )

    ax = Axis(
        fig[1:2, 1];
        limits=(nothing, (-14, 18)),
        title="Trend Inflation",
        xtickformat=dateticks,
    )

    lines!(fig[1:2, 1], vcat(0, data...); color=:red, linestyle=:dash)
    lines!(ax, zs[1, :]; color=:black)

    xs = mean(
        [hcat(getproperty.(path, :x)...) for path in all_paths],
        Weights(StatsBase.weights(states.log_weights))
    )

    ax1 = Axis(fig[1, 2]; title="Volatility", xtickformat=dateticks)
    lines!(ax1, exp.(0.5 * xs[1, :]); color=:black, label="permanent")
    axislegend(ax1; position=:rt)

    ax2 = Axis(fig[2, 2]; xtickformat=dateticks)
    lines!(ax2, exp.(0.5 * xs[2, :]); color=:black, label="transitory")
    axislegend(ax2; position=:lt)

    display(fig)

    return ll
end

rng = MersenneTwister(1234);
fred_data = [[pce] for pce in fred.data.value];

# plot both models side by side, notice the difference in volatility
plot_ucsv(rng, UCSV(0.2), fred_data);
plot_ucsv(rng, UCSVO(0.2, 0.05), fred_data);
