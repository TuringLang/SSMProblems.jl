using GeneralisedFilters
using Distributions

using Random
using StatsBase

const INFL_PATH = @__DIR__
include(joinpath(INFL_PATH, "utilities.jl"))

## STATE PRIORS ############################################################################

struct LocalLevelTrendPrior{T<:Real} <: StatePrior end

function GeneralisedFilters.distribution(prior::LocalLevelTrendPrior{T}) where {T}
    return product_distribution(
        Normal(zero(T), T(5)), Normal(zero(T), T(1)), Normal(zero(T), T(1))
    )
end

struct OutlierAdjustedTrendPrior{T<:Real} <: StatePrior end

function GeneralisedFilters.distribution(prior::OutlierAdjustedTrendPrior{T}) where {T}
    return product_distribution(
        Normal(zero(T), T(5)), Normal(zero(T), T(1)), Normal(zero(T), T(1)), Dirac(one(T))
    )
end

## LATENT DYNAMICS #########################################################################

struct LocalLevelTrend{ΓT<:AbstractVector} <: LatentDynamics
    γ::ΓT
end

function GeneralisedFilters.logdensity(proc::LocalLevelTrend, ::Integer, prev_state, state)
    vol_prob = sum(logpdf(Normal(prev_state[i + 1], proc.γ[i]), state[i + 1]) for i in 1:2)
    trend_prob = logpdf(Normal(prev_state[1], exp(state[2] / 2)), state[1])
    return vol_prob + trend_prob
end

function GeneralisedFilters.simulate(
    rng::AbstractRNG, proc::LocalLevelTrend, step::Integer, state::AbstractVector{T}
) where {T<:Real}
    new_state = deepcopy(state)
    new_state[2:3] += proc.γ .* randn(rng, T, 2)
    new_state[1] += exp(new_state[2] / 2) * randn(rng, T)
    return new_state
end

struct OutlierAdjustedTrend{ΓT<:AbstractVector} <: LatentDynamics
    trend::LocalLevelTrend{ΓT}
    switch_dist::Bernoulli
    outlier_dist::Uniform
end

function GeneralisedFilters.logdensity(
    proc::OutlierAdjustedTrend, step::Integer, prev_state, state
)
    base = GeneralisedFilters.logdensity(proc.trend, step, prev_state, state)
    p = succprob(proc.switch_dist)
    outlier = state[4] == 1 ? log1p(-p) : log(p) + logpdf(proc.outlier_dist, state[4])
    return base + outlier
end

function GeneralisedFilters.simulate(
    rng::AbstractRNG, proc::OutlierAdjustedTrend, step::Integer, state::AbstractVector{T}
) where {T<:Real}
    new_state = GeneralisedFilters.simulate(rng, proc.trend, step, state)
    new_state[4] = rand(rng, proc.switch_dist) ? rand(rng, proc.outlier_dist) : one(T)
    return new_state
end

## OBSERVATION PROCESS #####################################################################

struct OutlierAdjustedObservation <: ObservationProcess end

function GeneralisedFilters.distribution(
    proc::OutlierAdjustedObservation, step::Integer, state::AbstractVector
)
    return Normal(state[1], sqrt(state[4]) * exp(state[3] / 2))
end

struct SimpleObservation <: ObservationProcess end

function GeneralisedFilters.distribution(
    proc::SimpleObservation, step::Integer, state::AbstractVector
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
    states, ll, tree = filter_with_ancestry(rng, model, alg, data)

    fig = Figure(; size=(1200, 500), fontsize=16)
    dateticks = date_format(fred_data.date)

    all_paths = map(x -> hcat(x...), GeneralisedFilters.get_ancestry(tree))
    mean_paths = mean(all_paths, StatsBase.weights(states))

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
plot_ucsv(rng, UCSV(0.2), fred_data.value);
plot_ucsv(rng, UCSVO(0.2, 0.05), fred_data.value);
