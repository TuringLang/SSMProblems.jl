using CairoMakie

const ARRAY_COLOR = colorant"#2a78d6"
const SARRAY_COLOR = colorant"#eb6834"

const FILTER_STYLES = (
    ("Kalman filter", colorant"#2a78d6", :circle),
    ("Bootstrap filter", colorant"#eb6834", :rect),
    ("RBPF", colorant"#1baf7a", :utriangle),
)

function plot_filter_comparison(filters)
    n = nrow(filters)
    fig = Figure(; size=(700, 320))
    ax = Axis(
        fig[1, 1];
        xticks=(1:n, filters.filter),
        ylabel="median time (ms)",
        yscale=log10,
        title="Filtering 100 observations of a 4-dimensional state",
    )
    barplot!(
        ax,
        [1:n; 1:n],
        [filters.time_μs_Array; filters.time_μs_SArray] ./ 1e3;
        dodge=[fill(1, n); fill(2, n)],
        dodge_gap=0.06,
        color=[fill(ARRAY_COLOR, n); fill(SARRAY_COLOR, n)],
        # A log axis has no zero, so bars must start from a positive baseline.
        fillto=1e-3,
    )
    # With two dodged bars of the default width, the SArray bar is centred at i + 0.2.
    text!(
        ax,
        (1:n) .+ 0.2,
        filters.time_μs_SArray ./ 1e3;
        text=["$(round(s; digits=1))×" for s in filters.speedup],
        align=(:center, :bottom),
        offset=(0, 4),
    )
    Legend(
        fig[1, 2],
        [PolyElement(; color=ARRAY_COLOR), PolyElement(; color=SARRAY_COLOR)],
        ["Array", "SArray"],
    )
    return fig
end

function plot_dimension_sweep(comparison)
    Ds = sort(unique(comparison.D))
    fig = Figure(; size=(700, 360))
    ax = Axis(
        fig[1, 1];
        xlabel="state dimension",
        ylabel="speed-up of SArray over Array",
        xscale=log2,
        yscale=log10,
        yticks=([1, 2, 5, 10, 20], ["1×", "2×", "5×", "10×", "20×"]),
        xticks=Ds,
        title="Speed-up from static arrays by state dimension",
    )
    hlines!(ax, [1.0]; color=:gray, linestyle=:dash)
    for (name, color, marker) in FILTER_STYLES
        rows = comparison[comparison.filter .== name, :]
        scatterlines!(
            ax, rows.D, rows.speedup; color, marker, markersize=10, linewidth=2, label=name
        )
    end
    Legend(fig[1, 2], ax)
    return fig
end
