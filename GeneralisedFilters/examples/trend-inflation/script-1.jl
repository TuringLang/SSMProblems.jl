using GeneralisedFilters
using SSMProblems
using Distributions

using Random
using StatsBase

include("utilities.jl")

## STATE PRIORS ############################################################################

struct LocalLevelTrendPrior{T<:Real} <: StatePrior end

function SSMProblems.distribution(prior::LocalLevelTrendPrior{T}; kwargs...) where {T}
    return product_distribution(
        Normal(zero(T), T(5)), Normal(zero(T), T(1)), Normal(zero(T), T(1))
    )
end

struct OutlierAdjustedTrendPrior{T<:Real} <: StatePrior end

function SSMProblems.distribution(prior::OutlierAdjustedTrendPrior{T}; kwargs...) where {T}
    return product_distribution(
        Normal(zero(T), T(5)), Normal(zero(T), T(1)), Normal(zero(T), T(1)), Dirac(one(T))
    )
end

## LATENT DYNAMICS #########################################################################

struct LocalLevelTrend{ΓT<:AbstractVector} <: LatentDynamics
    γ::ΓT
end

function SSMProblems.logdensity(
    proc::LocalLevelTrend, ::Integer, prev_state, state, kwargs...
)
    vol_prob = logpdf(MvNormal(prev_state[2:end], proc.γ), state[2:end])
    trend_prob = logpdf(Normal(prev_state[1], exp(prev_state[2] / 2)), state[1])
    return vol_prob + trend_prob
end

function SSMProblems.simulate(
    rng::AbstractRNG,
    proc::LocalLevelTrend,
    step::Integer,
    state::AbstractVector{T};
    kwargs...,
) where {T<:Real}
    new_state = deepcopy(state)
    new_state[2:3] += proc.γ .* randn(rng, T, 2)
    new_state[1] += exp(new_state[2] / 2) * randn(T)
    return new_state
end

struct OutlierAdjustedTrend{ΓT<:AbstractVector} <: LatentDynamics
    trend::LocalLevelTrend{ΓT}
    switch_dist::Bernoulli
    outlier_dist::Uniform
end

function SSMProblems.logdensity(
    proc::OutlierAdjustedTrend, step::Integer, prev_state, state, kwargs...
)
    return SSMProblems.logdensity(proc.trend, step, prev_state, state; kwargs...)
end

function SSMProblems.simulate(
    rng::AbstractRNG,
    proc::OutlierAdjustedTrend,
    step::Integer,
    state::AbstractVector{T};
    kwargs...,
) where {T<:Real}
    new_state = SSMProblems.simulate(rng, proc.trend, step, state; kwargs...)
    new_state[4] = rand(rng, proc.switch_dist) ? rand(rng, proc.outlier_dist) : one(T)
    return new_state
end

## OBSERVATION PROCESS #####################################################################

struct OutlierAdjustedObservation <: ObservationProcess end

function SSMProblems.distribution(
    proc::OutlierAdjustedObservation, step::Integer, state::AbstractVector; kwargs...
)
    return Normal(state[1], sqrt(state[4]) * exp(state[3] / 2))
end

struct SimpleObservation <: ObservationProcess end

function SSMProblems.distribution(
    proc::SimpleObservation, step::Integer, state::AbstractVector; kwargs...
)
    return Normal(state[1], exp(state[3] / 2))
end

## MAIN ####################################################################################

# include UCSV as a baseline
function UCSV(γ::T) where {T<:Real}
    return StateSpaceModel(
        LocalLevelTrendPrior{T}(), LocalLevelTrend(fill(γ, 2)), SimpleObservation()
    )
end

# quick demo of the outlier-adjusted univariate UCSV model
function UCSVO(γ::T, prob::T) where {T<:Real}
    trend = LocalLevelTrend(fill(γ, 2))
    return StateSpaceModel(
        OutlierAdjustedTrendPrior{T}(),
        OutlierAdjustedTrend(trend, Bernoulli(prob), Uniform{T}(2, 10)),
        OutlierAdjustedObservation(),
    )
end

# wrapper to plot and demo the model
function plot_ucsv(rng::AbstractRNG, model, data)
    alg = BF(2^14; threshold=1.0, resampler=Systematic())
    sparse_ancestry = GeneralisedFilters.AncestorCallback(nothing)
    states, ll = GeneralisedFilters.filter(rng, model, alg, data; callback=sparse_ancestry)

    fig = Figure(; size=(1200, 500), fontsize=16)
    dateticks = date_format(fred.data.date)

    all_paths = map(x -> hcat(x...), GeneralisedFilters.get_ancestry(sparse_ancestry.tree))
    mean_paths = mean(all_paths, Weights(StatsBase.weights(states.log_weights)))

    ax = Axis(
        fig[1:2, 1];
        limits=(nothing, (-14, 18)),
        title="Trend Inflation",
        xtickformat=dateticks,
    )

    lines!(fig[1:2, 1], vcat(0, data...); color=:red, linestyle=:dash)
    lines!(ax, mean_paths[1, :]; color=:black)

    ax1 = Axis(fig[1, 2]; title="Volatility", xtickformat=dateticks)
    lines!(ax1, exp.(0.5 * mean_paths[2, :]); color=:black, label="permanent")
    axislegend(ax1; position=:rt)

    ax2 = Axis(fig[2, 2]; xtickformat=dateticks)
    lines!(ax2, exp.(0.5 * mean_paths[3, :]); color=:black, label="transitory")
    axislegend(ax2; position=:lt)

    display(fig)
    return ll
end

rng = MersenneTwister(1234);

# plot both models side by side, notice the difference in volatility
plot_ucsv(rng, UCSV(0.2), fred.data.value);
plot_ucsv(rng, UCSVO(0.2, 0.05), fred.data.value);
