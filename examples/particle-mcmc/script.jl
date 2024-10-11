using AdvancedMH
using CairoMakie

include("particles.jl")
include("resamplers.jl")
include("simple-filters.jl")

## DATA GENERATING PROCESS #################################################################

# use a local level trend model
function simulation_model(σx²::T, σy²::T) where {T<:Real}
    init = Gaussian(zeros(T, 2), PDMat(diagm(ones(T, 2))))
    dyn = LinearGaussianLatentDynamics(T[1 1; 0 1], T[0; 0], PDiagMat([σx²; 0]), init)
    obs = LinearGaussianObservationProcess(T[1 0], [σy²;;])
    return StateSpaceModel(dyn, obs)
end

# generate model and data
rng = MersenneTwister(1234);
true_params = randexp(rng, Float32, 2);
true_model = simulation_model(true_params...);
_, _, data = sample(rng, true_model, 150);

## FILTERING DEMONSTRATION #################################################################

filter = BF(1024; threshold=1.0, resampler=Systematic());
sparse_ancestry = AncestorCallback(eltype(true_model.dyn), filter.N, 1.0);
_, llbf = sample(rng, true_model, filter, data; callback=sparse_ancestry);

begin
    fig = Figure(; size=(600, 400))
    ax = Axis(fig[1, 1]; title="Surviving Lineage")

    # TODO: make the ancestry trace more palatable
    all_paths = map(x -> hcat(x...), get_ancestry(sparse_ancestry.tree))
    n_paths = length(all_paths)

    lines!.(ax, getindex.(all_paths, 1, :), color=(:black, maximum([2 / n_paths, 1e-2])))
    lines!(ax, vcat(data...); color=:red, linestyle=:dash)

    display(fig)
end

## COMPARING RESAMPLERS ####################################################################

function plot_resampler(alg, model, data)
    filter = BF(20; threshold=1.0, resampler=alg)
    resampler_ancestry = ResamplerCallback(20)

    rng = MersenneTwister(1234)
    sample(rng, model, filter, data; callback=resampler_ancestry)

    fig = Figure(; size=(600, 300))
    ax = Axis(
        fig[1, 1];
        xticks=0:10:50,
        yticks=0:5:20,
        limits=(nothing, (-5, 25)),
        title="$(typeof(alg))",
    )

    paths = get_ancestry(resampler_ancestry.tree)
    scatterlines!.(
        ax, paths, color=(:black, 0.25), markercolor=:black, markersize=5, linewidth=1
    )

    return fig
end

for rs in [Multinomial(), Systematic(), Metropolis(), Rejection()]
    fig = plot_resampler(rs, true_model, data[1:50])
    display(fig)
end

## PARTICLE MCMC ###########################################################################

# consider a default Gamma prior with Float32s
prior_dist = product_distribution(Gamma(1.0f0), Gamma(1.0f0));

# basic RWMH ala AdvancedMH
function density(θ::Vector{T}) where {T<:Real}
    if insupport(prior_dist, θ)
        # _, ll = sample(rng, simulation_model(θ...), BF(512), data)
        _, ll = sample(rng, simulation_model(θ...), KF(), data)
        return ll + logpdf(prior_dist, θ)
    else
        return -Inf
    end
end

pmmh = RWMH(MvNormal(zeros(Float32, 2), (0.01f0) * I));
model = DensityModel(density);

# works with AdvancedMH out of the box
chains = sample(model, pmmh, 50_000);
burn_in = 1_000;

# plot the posteriors
hist_plots = begin
    param_post = hcat(getproperty.(chains[burn_in:end], :params)...)
    fig = Figure(; size=(800, 300))

    for i in 1:2
        # plot the posteriors with burn-in
        hist(
            fig[1, i],
            param_post[i, :];
            color=(:black, 0.4),
            strokewidth=1,
            normalization=:pdf,
        )

        # plot the true values
        vlines!(fig[1, i], true_params[i]; color=:red, linestyle=:dash, linewidth=3)
    end

    fig
end

# this is useful for SMC algorithms like SMC² or density tempered SMC
acc_ratio = mean(getproperty.(chains, :accepted))
