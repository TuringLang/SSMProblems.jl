# # Trend Inflation
#
# Inflation measurements combine a persistent trend with short-lived fluctuations. This
# example estimates both components while allowing their variances to change over time,
# using an unobserved-components model with stochastic volatility based on Stock and Watson
# (2007, 2016).
#
# The trend follows a Gaussian random walk. Its innovation variance and the observation
# noise variance are controlled by two latent log variances. Conditional on those log
# variances, the trend model is linear and Gaussian. A Rao-Blackwellised particle filter
# therefore samples the log variances and uses a Kalman filter within each particle to
# integrate out the trend. This avoids sampling all three latent components together.

#nb # Install dependencies so the notebook runs on a fresh Colab runtime. GeneralisedFilters
#nb # and SSMProblems are added from the repo's main branch (registered versions may be too
#nb # old, e.g. lack ReferenceTrajectory):
#nb import Pkg, Downloads
#nb Downloads.download(
#nb     "https://raw.githubusercontent.com/TuringLang/SSMProblems.jl/main/GeneralisedFilters/examples/trend-inflation/Project.toml",
#nb     "Project.toml",
#nb )
#nb Pkg.activate(".")
#nb Pkg.add([
#nb     Pkg.PackageSpec(; url="https://github.com/TuringLang/SSMProblems.jl", subdir="GeneralisedFilters", rev="main"),
#nb     Pkg.PackageSpec(; url="https://github.com/TuringLang/SSMProblems.jl", subdir="SSMProblems", rev="main"),
#nb ])
#nb Pkg.instantiate()

using GeneralisedFilters
using Distributions
using Random
using StatsBase
using LinearAlgebra
using StaticArrays

const GF = GeneralisedFilters

INFL_PATH = joinpath(pkgdir(GeneralisedFilters), "examples", "trend-inflation"); #hide
include(joinpath(INFL_PATH, "utilities.jl")); #hide
#nb # Download the helper script and data so the notebook is self-contained on Colab:
#nb using Downloads
#nb INFL_URL = "https://raw.githubusercontent.com/TuringLang/SSMProblems.jl/main/GeneralisedFilters/examples/trend-inflation"
#nb for f in ("utilities.jl", "data.csv")
#nb     isfile(f) || Downloads.download("$(INFL_URL)/$(f)", f)
#nb end
#nb INFL_PATH = pwd()
#nb include(joinpath(INFL_PATH, "utilities.jl"))

# ## Model Definition

# Let $z_t$ denote trend inflation and $y_t$ observed inflation. The local level model is
#
# ```math
# \begin{aligned}
#     z_t &= z_{t-1} + \varepsilon_t, & \varepsilon_t &\sim N(0, \exp(h_{\varepsilon,t})), \\
#     y_t &= z_t + \eta_t, & \eta_t &\sim N(0, \exp(h_{\eta,t})).
# \end{aligned}
# ```
#
# The two log variances also follow random walks:
#
# ```math
# h_{j,t} = h_{j,t-1} + \gamma_j u_{j,t}, \qquad
# u_{j,t} \sim N(0,1), \quad j \in \{\varepsilon,\eta\}.
# ```
#
# In the implementation below, the outer state is
# $x_t = (h_{\varepsilon,t}, h_{\eta,t})$. The values in `γ` are the standard deviations
# of the log-variance innovations. The initial log variances have independent standard
# normal priors, and the initial trend has a $N(0,100)$ prior.
#
# #### Stochastic Volatility Process
#
# Define the prior and transition for the outer state using the process interface.
# Although the log variances themselves follow Gaussian random walks, their effect on the
# observations is nonlinear. This is the part of the model represented by particles.

struct StochasticVolatilityPrior{T<:Real} <: StatePrior end

# 

function GF.distribution(prior::StochasticVolatilityPrior{T}) where {T}
    return product_distribution(Normal(zero(T), T(1)), Normal(zero(T), T(1)))
end

# This bootstrap particle filter only needs to simulate the outer transition, so a
# `simulate` method is sufficient here. Algorithms that evaluate its density would also
# need a transition `logdensity` or `distribution` method.

struct StochasticVolatility{ΓT<:AbstractVector} <: LatentDynamics
    γ::ΓT
end

# 

function GF.simulate(
    rng::AbstractRNG, proc::StochasticVolatility, step::Integer, state::AbstractVector{T}
) where {T<:Real}
    new_state = deepcopy(state)
    new_state[1:2] += proc.γ .* randn(rng, T, 2)
    return new_state
end

# #### Local Level Trend Process
#
# Define the conditional trend transition and observation model as functions of the outer
# state. `ctx.x_new` holds the log variances at the end of a transition, while `ctx.x`
# holds them at the observation time. Each function returns a Gaussian atom containing
# the matrix, offset and covariance for that step. Static arrays suit this small inner
# model and let the Kalman filter keep its state in fixed-size storage.
function local_level(ctx)
    return LinearGaussianDynamics(
        @SMatrix([1.0;;]), @SVector([0.0]), SMatrix{1,1}(exp(ctx.x_new[1]))
    )
end
function simple_observation(ctx)
    return LinearGaussianObservation(
        @SMatrix([1.0;;]), @SVector([0.0]), SMatrix{1,1}(exp(ctx.x[2]))
    )
end

# ### Unobserved Components with Stochastic Volatility

# Combine the outer prior and transition with the inner Gaussian prior, transition and
# observation model. Both log variances use the same innovation standard deviation `γ`.

function UCSV(γ::T) where {T<:Real}
    stoch_vol_prior = StochasticVolatilityPrior{T}()
    stoch_vol_process = StochasticVolatility(fill(γ, 2))

    return StateSpaceModel(
        stoch_vol_prior,
        stoch_vol_process,
        GaussianPrior(@SVector([0.0]), @SMatrix([100.0;;])),
        local_level,
        simple_observation,
    )
end;

# Run the Rao-Blackwellised filter with 4096 bootstrap particles and a Kalman filter for
# each particle's trend distribution. The `filter_with_ancestry` helper records the
# particle ancestry for plotting.

rng = MersenneTwister(1234);
states, ll, tree = filter_with_ancestry(
    rng,
    UCSV(0.2),
    RBPF(BF(2^12), KalmanFilter()),
    [SVector(pce) for pce in fred_data.value],
);

# `get_ancestry` recovers the surviving paths through the particle tree. The `mean_path`
# helper averages them using the final particle weights. For the trend, it averages the
# Gaussian filtering means stored along those paths. This gives a useful plot of the
# inferred trend, but does not perform Gaussian backward smoothing within each path.
# Ancestral paths can also lose diversity at earlier times as resampling removes particles.

trends, volatilities = mean_path(GF.get_ancestry(tree), states);
plot_ucsv(trends[1, :], eachrow(volatilities), fred_data)

# #### Outlier Adjustments

# To allow occasional large measurement errors, add an independent variance multiplier
# $s_t$ to the observation model, following the outlier adjustment of Stock and Watson
# (2016):

# ```math
# \eta_{t} \sim N(0, s_{t} \exp(h_{\eta,t})) \quad \quad s_{t} \sim \begin{cases}
# U(2,10) & \text{ with probability } p \\
# \delta(1) & \text{ with probability } 1 - p
# \end{cases}
# ```

# Add the multiplier as a third component of the outer state. Its initial value is fixed
# at one with `Dirac(1)`. Subsequent values are drawn independently at each transition.

struct OutlierAdjustedVolatilityPrior{T<:Real} <: StatePrior end

# 

function GF.distribution(prior::OutlierAdjustedVolatilityPrior{T}) where {T}
    return product_distribution(Normal(zero(T), T(1)), Normal(zero(T), T(1)), Dirac(one(T)))
end

# The new transition contains the original log-variance process and the two distributions
# used to draw the multiplier.

struct OutlierAdjustedVolatility{ΓT} <: LatentDynamics
    volatility::StochasticVolatility{ΓT}
    switch_dist::Bernoulli
    outlier_dist::Uniform
end

# Simulate the log variances, then replace the third component with the new multiplier.

function GF.simulate(
    rng::AbstractRNG,
    proc::OutlierAdjustedVolatility,
    step::Integer,
    state::AbstractVector{T},
) where {T<:Real}
    new_state = GF.simulate(rng, proc.volatility, step, state)
    new_state[3] = rand(rng, proc.switch_dist) ? rand(rng, proc.outlier_dist) : one(T)
    return new_state
end

# Multiply the observation variance by the third component. Conditional on the complete
# outer state, the observation model is still Gaussian.

function outlier_observation(ctx)
    return LinearGaussianObservation(
        @SMatrix([1.0;;]), @SVector([0.0]), SMatrix{1,1}(ctx.x[3] * exp(ctx.x[2]))
    )
end

# ### Outlier Adjusted UCSV

# Assemble the model with the new outer process and observation function. The inner prior
# and trend transition are unchanged.

function UCSVO(γ::T, prob::T) where {T<:Real}
    stoch_vol_prior = OutlierAdjustedVolatilityPrior{T}()
    stoch_vol_process = OutlierAdjustedVolatility(
        StochasticVolatility(fill(γ, 2)), Bernoulli(prob), Uniform{T}(2, 10)
    )

    return StateSpaceModel(
        stoch_vol_prior,
        stoch_vol_process,
        GaussianPrior(@SVector([0.0]), @SMatrix([100.0;;])),
        local_level,
        outlier_observation,
    )
end;

# Repeat the filtering calculation with an outlier probability of $p = 0.05$.

rng = MersenneTwister(1234);
states, ll, tree = filter_with_ancestry(
    rng,
    UCSVO(0.2, 0.05),
    RBPF(BF(2^12), KalmanFilter()),
    [SVector(pce) for pce in fred_data.value],
);

# Plot the trend and the two baseline volatility components as before. The third component
# contains the variance multipliers and is omitted from the plots. Large measurement errors
# can now be explained by a temporary multiplier as well as a change in baseline volatility.

trends, volatilities = mean_path(GF.get_ancestry(tree), states);
plot_ucsv(trends[1, :], eachrow(volatilities), fred_data)
