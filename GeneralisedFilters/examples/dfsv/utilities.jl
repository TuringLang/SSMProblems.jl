using CSV, DataFrames, Dates, Statistics, CairoMakie

const DATA_PATH = joinpath(@__DIR__, "5_Industry_Portfolios.csv")
const INDUSTRIES = ["Cnsmr", "Manuf", "HiTec", "Hlth", "Other"]

# Parse the Ken French 5 Industry Portfolios CSV. The file has multiple sections
# separated by blank lines; we want the first (monthly value-weighted returns).
function load_industry_data(; date_from=198501, date_to=202412)
    df = CSV.read(
        DATA_PATH,
        DataFrame;
        header=["date", INDUSTRIES...],
        skipto=13,
        limit=1200,
        missingstring=["-99.99", "-999"],
        types=Dict(:date => Int, (Symbol(i) => Float64 for i in INDUSTRIES)...),
        silencewarnings=true,
    )
    dropmissing!(df)
    filter!(row -> date_from <= row.date <= date_to, df)
    return df
end

function _yyyymm_to_date(yyyymm::Integer)
    y, m = divrem(yyyymm, 100)
    return Date(y, m, 1)
end

function plot_returns(df)
    dates = _yyyymm_to_date.(df.date)
    fig = Figure(; size=(1100, 500), fontsize=14)
    ax = Axis(
        fig[1, 1];
        title="Monthly Value-Weighted Returns — French 5 Industry Portfolios",
        xlabel="Date",
    )
    colors = Makie.wong_colors()
    for (i, ind) in enumerate(INDUSTRIES)
        lines!(ax, dates, df[!, ind]; label=ind, color=colors[i], linewidth=0.8)
    end
    axislegend(ax; position=:rt)
    return fig
end

function plot_volatilities(vol_paths, dates)
    T = length(dates)
    # vol_paths: Vector of length T, each element a vector of SVector{6} particle states
    # Each outer state is [g, u₁, ..., u₅]

    fig = Figure(; size=(1100, 700), fontsize=13)

    # Common volatility factor — full width, top row
    ax_g = Axis(fig[1, 1]; title="Common Volatility Factor gₜ", xlabel="Date")
    g_mean = [mean(getindex.(vol_paths[t], 1)) for t in 1:T]
    lines!(ax_g, dates, g_mean; color=:black)

    # Idiosyncratic log-vols in a nested 2×3 grid so they don't share column
    # boundaries with ax_g above.
    industry_grid = fig[2, 1] = GridLayout()
    colors = Makie.wong_colors()
    for (i, ind) in enumerate(INDUSTRIES)
        row = 1 + div(i - 1, 3)
        col = (i - 1) % 3 + 1
        ax = Axis(industry_grid[row, col]; title="$ind log-vol u_{$(i),t}", xlabel="Date")
        u_mean = [mean(getindex.(vol_paths[t], i + 1)) for t in 1:T]
        lines!(ax, dates, u_mean; color=colors[i])
    end

    return fig
end

function plot_chains(chain; burnin=0)
    param_names = string.(names(chain, :parameters))
    n_iter = size(chain, 1) - burnin
    fig = Figure(; size=(1100, 80 * length(param_names)), fontsize=11)
    for (i, pname) in enumerate(param_names)
        ax = Axis(fig[i, 1]; ylabel=pname)
        samples = Array(chain[(burnin + 1):end, pname, 1])
        lines!(ax, 1:n_iter, samples; linewidth=0.6)
    end
    return fig
end
