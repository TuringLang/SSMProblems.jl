# # Trend Inflation
#
# This example is a replication of the univariate state space model suggested by (Stock &
# Watson, 2016) using GeneralisedFilters to define a heirarchical model for use in Rao-
# Blackwellised particle filtering.

using GeneralisedFilters
using SSMProblems
using Distributions
using Random
using StatsBase
using LinearAlgebra

const GF = GeneralisedFilters

INFL_PATH = joinpath(@__DIR__, "..", "..", "..", "examples", "trend-inflation"); #hide
include(joinpath(INFL_PATH, "utilities.jl")); #hide

# ## Model Definition

# We begin by defining the local level trend model, a linear Gaussian model with a weakly
# stationary random walk component. The dynamics of which are as follows:

# ```math
# \begin{aligned}
#     y_{t} &= x_{t} + \eta_{t} \\
#     x_{t+1} &= x_{t} + \varepsilon_{t}
# \end{aligned}
# ```

# However, this model is not enough to capture trend dynamics when faced with structural
# breaks. (Stock & Watson, 2007) suggest adding a stochastic volatiltiy component, defined
# like so:

# ```math
# \begin{aligned}
#     \log \sigma_{\eta, t+1} = \log \sigma_{\eta, t} + \nu_{\eta, t} \\
#     \log \sigma_{\varepsilon, t+1} = \log \sigma_{\varepsilon, t} + \nu_{\varepsilon, t}
# \end{aligned}
# ```

# where $\nu_{z,t} \sim N(0, \gamma)$ for $z \in \{ \varepsilon, \eta \}$.

# Using `GeneralisedFilters`, we can construct a heirarchical version of this model such
# that the local level trend component is conditionally linear Gaussian on the volatility
# draws.

# #### Stochastic Volatility Process

# We begin by defining the non-linear dynamics, which aren't conditioned contemporaneous
# states. Since these processes are traditionally non-linear/non-Gaussian we use the
# SSMProblems interface to define the stochastic volatility components.

struct StochasticVolatilityPrior{T<:Real} <: StatePrior end

# 

function SSMProblems.distribution(prior::StochasticVolatilityPrior{T}; kwargs...) where {T}
    return product_distribution(Normal(zero(T), T(1)), Normal(zero(T), T(1)))
end

# For the dynamics, instead of using the `SSMProblems.distribution` utility, we only define
# the `simulate` method, which is sufficient for the RBPF.

struct StochasticVolatility{ΓT<:AbstractVector} <: LatentDynamics
    γ::ΓT
end

# 

function SSMProblems.simulate(
    rng::AbstractRNG,
    proc::StochasticVolatility,
    step::Integer,
    state::AbstractVector{T};
    kwargs...,
) where {T<:Real}
    new_state = deepcopy(state)
    new_state[1:2] += proc.γ .* randn(rng, T, 2)
    return new_state
end

# #### Local Level Trend Process
#
# For the conditionally linear and Gaussian components, we subtype the model and provide a
# keyword argument as the conditional element. In this case $A$ and $b$ remain constant, but
# $Q$ is conditional on the log variance, stored in `new_outer` (the nomenclature chosen for
# heirarchical modeling).

struct LocalLevelTrend <: LinearGaussianLatentDynamics end

# 

GF.calc_A(::LocalLevelTrend, ::Integer; kwargs...) = [1;;]
GF.calc_b(::LocalLevelTrend, ::Integer; kwargs...) = [0;]
function GF.calc_Q(::LocalLevelTrend, ::Integer; new_outer, kwargs...)
    return [exp(new_outer[1]);;]
end

# Similarly, we define the observation process conditional on a separate log variance.

struct SimpleObservation <: LinearGaussianObservationProcess end

# 

GF.calc_H(::SimpleObservation, ::Integer; kwargs...) = [1;;]
GF.calc_c(::SimpleObservation, ::Integer; kwargs...) = [0;]
function GF.calc_R(::SimpleObservation, ::Integer; new_outer, kwargs...)
    return [exp(new_outer[2]);;]
end

# ### Unobserved Components with Stochastic Volatility

# The state space model suggested by (Stock & Watson, 2007) can be constructed with the
# following method:

function UCSV(γ::T) where {T<:Real}
    stoch_vol_prior = StochasticVolatilityPrior{T}()
    stoch_vol_process = StochasticVolatility(fill(γ, 2))

    local_level_model = StateSpaceModel(
        GF.HomogeneousGaussianPrior(zeros(T, 1), Matrix(100.0I(1))),
        LocalLevelTrend(),
        SimpleObservation(),
    )

    return HierarchicalSSM(stoch_vol_prior, stoch_vol_process, local_level_model)
end;

# For plotting, we can extract the ancestry of the Rao Blackwellised particles using the
# callback system. For our inflation data, this reduces to the following:

rng = MersenneTwister(1234);
sparse_ancestry = GF.AncestorCallback(nothing);
states, ll = GF.filter(
    rng,
    UCSV(0.2),
    RBPF(KalmanFilter(), 2^12; threshold=1.0),
    [[pce] for pce in fred_data.value];
    callback=sparse_ancestry,
);

# The `sparse_ancestry` object stores a sparse ancestry tree which we can use to approximate
# the smoothed series without an additional backwards pass. We can convert this data
# structure to a human readable array by using `GeneralisedFilters.get_ancestry` and then
# take the mean path by passing a custom function.

trends, volatilities = mean_path(GF.get_ancestry(sparse_ancestry.tree), states);
plot_ucsv(trends[1, :], eachrow(volatilities), fred_data)

# #### Outlier Adjustments

# For additional robustness, (Stock & Watson, 2016) account for one-time measurement shocks
# and suggest an alteration in the observation equation, where

# ```math
# \eta_{t} \sim N(0, s_{t} \cdot \sigma_{\eta, t}^2) \quad \quad s_{t} \sim \begin{cases}
# U(0,2) & \text{ with probability } p \\
# \delta(1) & \text{ with probability } 1 - p
# \end{cases}
# ```

# The prior is the same as before, but with additional state which we can assume will always
# be 1; using the `Distributions` interface this is just `Dirac(1)`

struct OutlierAdjustedVolatilityPrior{T<:Real} <: StatePrior end

# 

function SSMProblems.distribution(
    prior::OutlierAdjustedVolatilityPrior{T}; kwargs...
) where {T}
    return product_distribution(Normal(zero(T), T(1)), Normal(zero(T), T(1)), Dirac(one(T)))
end

# In terms of the model definition, we can construct a separate `LatentDynamics` which
# contains the same volatility process as before, but with the respective draw in the third
# component.

struct OutlierAdjustedVolatility{ΓT} <: LatentDynamics
    volatility::StochasticVolatility{ΓT}
    switch_dist::Bernoulli
    outlier_dist::Uniform
end

# The simulation then calls the volatility process, and computes the outlier term in the
# third state

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

# For the observation process, we define a new object where $R$ is dependent on both the
# measurement volatility as well as this outlier adjustment coefficient.

struct OutlierAdjustedObservation <: LinearGaussianObservationProcess end

# 

GF.calc_H(::OutlierAdjustedObservation, ::Integer; kwargs...) = [1;;]
GF.calc_c(::OutlierAdjustedObservation, ::Integer; kwargs...) = [0;]
function GF.calc_R(::OutlierAdjustedObservation, ::Integer; new_outer, kwargs...)
    return [new_outer[3] * exp(new_outer[2]);;]
end

# ### Outlier Adjusted UCSV

# The state space model suggested by (Stock & Watson, 2007) can be constructed with the
# following method:

function UCSVO(γ::T, prob::T) where {T<:Real}
    stoch_vol_prior = OutlierAdjustedVolatilityPrior{T}()
    stoch_vol_process = OutlierAdjustedVolatility(
        StochasticVolatility(fill(γ, 2)), Bernoulli(prob), Uniform{T}(2, 10)
    )

    local_level_model = StateSpaceModel(
        GF.HomogeneousGaussianPrior(zeros(T, 1), Matrix(100.0I(1))),
        LocalLevelTrend(),
        OutlierAdjustedObservation(),
    )

    return HierarchicalSSM(stoch_vol_prior, stoch_vol_process, local_level_model)
end;

# We then repeat the same experiment, this time with an outlier probability of $p = 0.05$

rng = MersenneTwister(1234);
sparse_ancestry = GF.AncestorCallback(nothing)
states, ll = GF.filter(
    rng,
    UCSVO(0.2, 0.05),
    RBPF(KalmanFilter(), 2^12; threshold=1.0),
    [[pce] for pce in fred_data.value];
    callback=sparse_ancestry,
);

# this process is identical to the last, except with an additional `volatilities` state
# which captures the outlier distance. We omit this feature in the plots, but the impact is
# clear when comparing the maximum transitory noise around the GFC.

trends, volatilities = mean_path(GF.get_ancestry(sparse_ancestry.tree), states);
plot_ucsv(trends[1, :], eachrow(volatilities), fred_data)
