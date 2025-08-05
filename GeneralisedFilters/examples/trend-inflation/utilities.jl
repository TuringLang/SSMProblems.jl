using CSV, DataFrames
using CairoMakie
using Dates

fred_data = CSV.read(joinpath(INFL_PATH, "data.csv"), DataFrame)

## PLOTTING UTILITIES ######################################################################

function _mean_path(f, paths, weights)
    return mean(map(x -> hcat(f(x)...), paths), StatsBase.weights(weights))
end

# for normal collections
mean_path(paths, weights) = _mean_path(identity, paths, weights)

# for rao blackwellised particles
function mean_path(
    paths::Vector{Vector{T}}, weights
) where {T<:GeneralisedFilters.RaoBlackwellisedParticle}
    zs = _mean_path(z -> getproperty.(getproperty.(z, :z), :μ), paths, weights)
    xs = _mean_path(x -> getproperty.(x, :x), paths, weights)
    return zs, xs
end

function plot_ucsv(trend, volatilities, fred_data)
    fig = Figure(; size=(1200, 500), fontsize=16)
    dateticks = date_format(fred_data.date)

    trend_ax = Axis(
        fig[1:2, 1];
        limits=(nothing, (-14, 18)),
        title="Trend Inflation",
        xtickformat=dateticks,
    )

    lines!(fig[1:2, 1], vcat(0, fred_data.value...); color=:red, linestyle=:dash)
    lines!(trend_ax, trend; color=:black)

    vol_ax_1 = Axis(fig[1, 2]; title="Volatility", xtickformat=dateticks)
    lines!(vol_ax_1, exp.(0.5 * volatilities[1]); color=:black, label="permanent")
    axislegend(vol_ax_1; position=:rt)

    vol_ax_2 = Axis(fig[2, 2]; xtickformat=dateticks)
    lines!(vol_ax_2, exp.(0.5 * volatilities[2]); color=:black, label="transitory")
    axislegend(vol_ax_2; position=:lt)

    return fig
end

# this is essential for plotting dates
date_format(dates) = x -> [Dates.format(dates[floor(Int, i) + 1], "yyyy") for i in x]
