# # Speeding Up Filters with Static Arrays
#
# For models with a small state dimension, much of the time spent filtering goes into
# managing memory rather than doing arithmetic. An ordinary `Vector` or `Matrix` lives on
# the heap, so every intermediate mean, covariance and particle state requires an
# allocation that the garbage collector must later clean up. A particle filter with
# thousands of particles makes thousands of these allocations at every time step.
#
# [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) provides arrays whose
# size is part of their type. The compiler can then keep them on the stack and unroll
# operations on them, removing both the allocations and most of the loop overhead.
# GeneralisedFilters preserves static array types throughout its algorithms, so switching
# a model over requires no changes to the filtering code itself.
#
# This example builds the same model with ordinary and static arrays and compares their
# performance for a Kalman filter, a bootstrap particle filter and a Rao-Blackwellised
# particle filter. It then shows how the benefit changes with the state dimension.

#nb # Install dependencies so the notebook runs on a fresh Colab runtime. GeneralisedFilters
#nb # and SSMProblems are added from the repo's main branch (registered versions may be too
#nb # old):
#nb import Pkg, Downloads
#nb Downloads.download(
#nb     "https://raw.githubusercontent.com/TuringLang/SSMProblems.jl/main/GeneralisedFilters/examples/static-arrays/Project.toml",
#nb     "Project.toml",
#nb )
#nb Pkg.activate(".")
#nb Pkg.add([
#nb     Pkg.PackageSpec(; url="https://github.com/TuringLang/SSMProblems.jl", subdir="GeneralisedFilters", rev="main"),
#nb     Pkg.PackageSpec(; url="https://github.com/TuringLang/SSMProblems.jl", subdir="SSMProblems", rev="main"),
#nb ])
#nb Pkg.instantiate()

using GeneralisedFilters
using StaticArrays
using BenchmarkTools
using CSV
using DataFrames
using Distributions
using LinearAlgebra
using Random

const GF = GeneralisedFilters

EXAMPLE_PATH = joinpath(pkgdir(GeneralisedFilters), "examples", "static-arrays"); #hide
include(joinpath(EXAMPLE_PATH, "plotting.jl")); #hide
#nb # Download the plotting helpers and precomputed benchmark results so the notebook is
#nb # self-contained on Colab:
#nb EXAMPLE_URL = "https://raw.githubusercontent.com/TuringLang/SSMProblems.jl/main/GeneralisedFilters/examples/static-arrays"
#nb for f in ("plotting.jl", "results.csv")
#nb     isfile(f) || Downloads.download("$(EXAMPLE_URL)/$(f)", f)
#nb end
#nb EXAMPLE_PATH = pwd()
#nb include(joinpath(EXAMPLE_PATH, "plotting.jl"))
RESULTS_PATH = joinpath(EXAMPLE_PATH, "results.csv")

# ## A Constant-Velocity Tracking Model
#
# We track an object moving through $k$-dimensional space. Its state $z_t \in
# \mathbb{R}^{2k}$ stacks its position $p_t$ and velocity $v_t$, and we observe its position
# with Gaussian noise. With a time step $\Delta t$, the velocity follows a random walk and
# the position integrates the velocity:
#
# ```math
# \begin{aligned}
#     z_t &= \begin{pmatrix} I_k & \Delta t I_k \\ 0 & I_k \end{pmatrix} z_{t-1} + w_t,
#     & w_t &\sim \mathcal{N}(0, Q), \\
#     y_t &= \begin{pmatrix} I_k & 0 \end{pmatrix} z_t + \varepsilon_t,
#     & \varepsilon_t &\sim \mathcal{N}(0, r I_k),
# \end{aligned}
# ```
#
# where $Q$ is the covariance of a continuous-time white-noise acceleration integrated
# over one time step.
#
# The following function returns the model parameters as ordinary Julia arrays.

function tracking_parameters(k; Δt=0.1, q=1.0, r=0.5)
    Iₖ = Matrix(1.0I, k, k)
    return (;
        μ0=zeros(2k),
        Σ0=Matrix(1.0I, 2k, 2k),
        A=kron([1.0 Δt; 0.0 1.0], Iₖ),
        b=zeros(2k),
        Q=kron(q * [Δt^3/3 Δt^2/2; Δt^2/2 Δt], Iₖ),
        H=kron([1.0 0.0], Iₖ),
        c=zeros(k),
        R=r * Iₖ,
    )
end

# To obtain the static version, we convert each parameter to an `SVector` or `SMatrix`.
# The size is read from the array at run time here, which is not type stable, but that
# only affects this one-off construction. Every array the filter subsequently creates has
# its size fixed in its type.

to_static(x::AbstractVector) = SVector{length(x)}(x)
to_static(X::AbstractMatrix) = SMatrix{size(X)...}(X)

function tracking_model(params)
    return StateSpaceModel(
        GaussianPrior(params.μ0, params.Σ0),
        LinearGaussianDynamics(params.A, params.b, params.Q),
        LinearGaussianObservation(params.H, params.c, params.R),
    )
end

# We start with an object moving in the plane, $k = 2$, giving a four-dimensional state.
# Both models share the same simulated observations, which we also convert to `SVector`s
# for the static model.

params = tracking_parameters(2)
array_model = tracking_model(params)
sarray_model = tracking_model(map(to_static, params))

T = 100
_, _, array_ys = simulate(Xoshiro(1), array_model, T)
sarray_ys = to_static.(array_ys)
typeof(array_ys[1]), typeof(sarray_ys[1])

# The two representations describe the same model, so the Kalman filter gives the same
# answer for each. Only the type of the returned state differs.

array_state, array_ll = GF.filter(array_model, KF(), array_ys)
sarray_state, sarray_ll = GF.filter(sarray_model, KF(), sarray_ys)
@assert array_ll ≈ sarray_ll
@assert mean(array_state) ≈ mean(sarray_state)
typeof(mean(sarray_state)), typeof(cov(sarray_state))

# This check is worth repeating for your own models. When an operation combines a static
# array with an ordinary one, the result is usually an ordinary array: an `SMatrix` times a
# `Matrix` is a `Matrix`. If one model component, such as the observation matrix or the
# observations themselves, were left as an ordinary array, the filter would still run and
# give the same answer, but its states would silently be promoted to `Array`s and much of
# the speed-up would be lost. Confirming that the filtered or smoothed state is still made
# of static arrays is a quick way to rule this out.

# ## A Rao-Blackwellised Model
#
# Static arrays save the most work when a filter creates many small arrays. A
# Rao-Blackwellised particle filter is an extreme case, since every particle carries its
# own Gaussian mean and covariance.
#
# To give the RBPF something to do, suppose the object manoeuvres unpredictably, so that
# the scale of its process noise changes over time. Let the outer state $x_t$ be the log of
# this scale, following an autoregressive process. Conditional on $x_t$, the tracking model
# is linear and Gaussian with process noise covariance $\exp(x_t) Q$, so each particle can
# integrate out the position and velocity with a Kalman filter.

function manoeuvring_model(params)
    return StateSpaceModel(
        DistributionPrior(Normal()),
        DistributionDynamics((t, x) -> Normal(0.9x, 0.5)),
        GaussianPrior(params.μ0, params.Σ0),
        ctx -> LinearGaussianDynamics(params.A, params.b, exp(ctx.x_new) * params.Q),
        LinearGaussianObservation(params.H, params.c, params.R),
    )
end

array_rb_model = manoeuvring_model(params)
sarray_rb_model = manoeuvring_model(map(to_static, params));

# Multiplying an `SMatrix` by a scalar returns another `SMatrix`, so the inner dynamics stay
# static without any extra effort. Take care when building matrices inside such functions:
# constructing an ordinary `Matrix` there would promote every particle's inner state to
# `Array`s.
#
# With a shared random seed, the particle filters draw the same random numbers for both
# representations, so they produce identical estimates.

N = 1000
algorithms = (bootstrap=BF(N), rbpf=RBPF(BF(N), KF()))
_, array_bf_ll = GF.filter(Xoshiro(2), array_model, algorithms.bootstrap, array_ys)
_, sarray_bf_ll = GF.filter(Xoshiro(2), sarray_model, algorithms.bootstrap, sarray_ys)
_, array_rb_ll = GF.filter(Xoshiro(2), array_rb_model, algorithms.rbpf, array_ys)
_, sarray_rb_ll = GF.filter(Xoshiro(2), sarray_rb_model, algorithms.rbpf, sarray_ys)
@assert array_bf_ll ≈ sarray_bf_ll
@assert array_rb_ll ≈ sarray_rb_ll

# ## Benchmarking
#
# We time each filter with BenchmarkTools, resetting the random number generator before
# every sample so that each run does the same work. The reported time is the median over
# samples; the memory and allocation counts are for a single run.

function benchmark_filter(model, algo, ys)
    trial = @benchmark GF.filter(rng, $model, $algo, $ys) setup = (rng = Xoshiro(2))
    return (;
        time_μs=median(trial).time / 1e3, memory_bytes=trial.memory, allocs=trial.allocs
    )
end

# We benchmark all three filters on the tracking problem for several spatial dimensions
# $k$, giving $2k$-dimensional states. Before benchmarking each model, we also time its
# first call to the filter, which includes compilation.

function benchmark_dimension!(rows, k)
    params_k = tracking_parameters(k)
    _, _, ys = simulate(Xoshiro(1), tracking_model(params_k), T)
    representations = (
        ("Array", params_k, ys), ("SArray", map(to_static, params_k), to_static.(ys))
    )
    for (array_type, p, obs) in representations
        filters = (
            ("Kalman filter", tracking_model(p), KF()),
            ("Bootstrap filter", tracking_model(p), algorithms.bootstrap),
            ("RBPF", manoeuvring_model(p), algorithms.rbpf),
        )
        for (name, model, algo) in filters
            first_call_s = @elapsed GF.filter(Xoshiro(2), model, algo, obs)
            bench = benchmark_filter(model, algo, obs)
            push!(rows, (; filter=name, D=2k, array_type, first_call_s, bench...))
        end
    end
    return rows
end

run_benchmarks() = DataFrame(foldl(benchmark_dimension!, (1, 2, 3, 5, 8, 12, 16); init=[]))

# Benchmarks are noisy and slow, so rather than run them while building this page, we load
# results computed ahead of time by `run_benchmarks()`. They were generated with Julia
# 1.12.7 using one Julia thread on an AMD Ryzen Threadripper 7960X. Timings will differ on
# other machines, but the relative speed-ups should be broadly similar.

# Regenerate with `REGENERATE_BENCHMARKS=true julia --project script.jl`. #src
if get(ENV, "REGENERATE_BENCHMARKS", "false") == "true" #src
    CSV.write(RESULTS_PATH, run_benchmarks()) #src
end #src
results = CSV.read(RESULTS_PATH, DataFrame)

function speedups(results)
    arrays = results[results.array_type .== "Array", :]
    sarrays = results[results.array_type .== "SArray", :]
    joined = innerjoin(arrays, sarrays; on=[:filter, :D], renamecols="_Array" => "_SArray")
    joined.speedup = joined.time_μs_Array ./ joined.time_μs_SArray
    return sort(joined, :D)
end

comparison = speedups(results)
planar = comparison[comparison.D .== 4, :]
select(
    planar,
    :filter,
    :time_μs_Array => ByRow(t -> round(t; sigdigits=3)) => "Array time (μs)",
    :time_μs_SArray => ByRow(t -> round(t; sigdigits=3)) => "SArray time (μs)",
    :speedup => ByRow(s -> round(s; digits=1)) => "speed-up",
    :memory_bytes_Array => ByRow(m -> round(m / 2^20; sigdigits=3)) => "Array memory (MiB)",
    :memory_bytes_SArray =>
        ByRow(m -> round(m / 2^20; sigdigits=3)) => "SArray memory (MiB)",
    :allocs_Array => "Array allocations",
    :allocs_SArray => "SArray allocations",
)

# For the planar problem, all three filters run around ten times faster with static
# arrays. The static Kalman filter does not allocate at all. The particle filters still
# allocate their particle vectors and resampling indices at each step, but no longer
# allocate separate arrays for each particle. The RBPF saves the most time in absolute
# terms, because static arrays remove the allocation of a mean and a covariance for every
# particle at every time step.

plot_filter_comparison(planar)

# ## How the Benefit Scales with Dimension
#
# Static arrays are not a free lunch. Their operations are unrolled at compile time, so the
# compiled code grows rapidly with the array size, and beyond a certain size the optimised
# BLAS routines used for ordinary matrices win.

plot_dimension_sweep(comparison)

# For the smallest states, static arrays make every filter between 15 and 20 times faster.
# The advantage shrinks as the state grows, and by 32 dimensions it has all but gone for
# the Kalman and bootstrap filters. The RBPF keeps a threefold speed-up even there, because
# it still avoids allocating a mean and covariance for each of its particles.
#
# Part of this drop-off is by design, since StaticArrays only uses its specialised
# implementations for small matrices. Several routines switch to more general code beyond
# $14 \times 14$, including
# [LU factorisation](https://github.com/JuliaArrays/StaticArrays.jl/blob/v1.9.22/src/lu.jl#L74)
# and therefore
# [linear solves](https://github.com/JuliaArrays/StaticArrays.jl/blob/v1.9.22/src/solve.jl#L58)
# and [inverses](https://github.com/JuliaArrays/StaticArrays.jl/blob/v1.9.22/src/inv.jl#L76).
# [Cholesky factorisation](https://github.com/JuliaArrays/StaticArrays.jl/blob/v1.9.22/src/cholesky.jl#L26)
# is unrolled up to $24 \times 24$ and beyond that copies its input to an ordinary
# `Matrix`. This is why the static bootstrap filter, which draws each particle from a
# Gaussian with a $32 \times 32$ covariance, starts allocating again at 32 dimensions. The
# StaticArrays documentation suggests, as a rough rule of thumb, using ordinary arrays for
# anything larger than about 100 elements.
#
# Each array size is a different type, so the static filters are also recompiled for every
# dimension, whereas `Array` code is compiled once and reused. The table below shows how
# much longer the first calls to the three static filters take in total than compiled
# runs. (The four-dimensional filters were already compiled above, so they are omitted.)
# This cost is paid once per session, so it is rarely significant for the long-running
# inference where static arrays are most useful.

sweep = comparison[comparison.D .!= 4, :]
sweep.compile_s = @. sweep.first_call_s_SArray - sweep.time_μs_SArray / 1e6
combine(
    groupby(sweep, :D),
    :compile_s => (s -> round(sum(s); sigdigits=2)) => "SArray compilation (s)",
)

# ## Practical Advice
#
# - Static arrays pay off most for small states. The absolute savings are largest for
#   particle filters and RBPFs, where many small arrays are created at each step.
# - Make every part of the model static, including the observations. Mixing static and
#   ordinary arrays usually produces ordinary arrays, losing the benefit. Check the type of
#   the filtered or smoothed state to confirm that it is still static.
# - Functions that construct model components, such as the inner dynamics of a
#   hierarchical model, should also return static arrays. `SA[...]`, `@SMatrix` and
#   arithmetic on existing static arrays all preserve static types.
# - The array size must be known when the code is compiled. If the dimension is only known
#   at run time or varies between calls, ordinary arrays are the better choice.
# - For larger states, ordinary arrays are usually the better choice. Beyond a few tens of
#   dimensions static arrays compile slowly and, unless the filter creates many small
#   arrays as an RBPF does, run little faster.
