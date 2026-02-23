# # Dynamic Factor Stochastic Volatility
#
# This example demonstrates Rao-Blackwellised Particle Gibbs (RBPG) with ancestor
# sampling and NUTS parameter updates, applied to a multivariate Dynamic Factor
# Stochastic Volatility (DFSV) model for monthly US industry portfolio returns.
#
# The model decomposes return volatility into:
# - A **common volatility factor** $g_t$ capturing economy-wide variance regimes, and
# - **Idiosyncratic stochastic volatility** $u_{i,t}$ per industry.
#
# Latent **linear factors** $f_t \in \mathbb{R}^2$ drive cross-industry return
# correlations and are integrated out analytically via Rao-Blackwellisation,
# greatly reducing the effective particle dimension.

using GeneralisedFilters
using SSMProblems
using Distributions
using DistributionsAD
using PDMats
using StaticArrays
using LinearAlgebra
using Random
using Statistics

using AbstractMCMC: AbstractMCMC
using AdvancedHMC: NUTS, HMC
using ADTypes: ADTypes
using MCMCChains: MCMCChains
using Turing: @model
using Zygote
import ChainRulesCore: ChainRulesCore, NoTangent

const GF = GeneralisedFilters

DFSV_PATH = joinpath(@__DIR__, "..", "..", "..", "GeneralisedFilters", "examples", "dfsv"); #hide
# DFSV_PATH = joinpath(@__DIR__)
includet(joinpath(DFSV_PATH, "utilities.jl")); #hide

# ## Model
#
# Let $y_t \in \mathbb{R}^5$ denote monthly (demeaned) log-returns. The state
# vector splits into two components:
#
# **Nonlinear (particle) state**: $x_t = [g_t,\, u_{1:5,t}] \in \mathbb{R}^6$
# - $g_t$: common log-volatility factor, $\text{AR}(1)$ with mean zero
# - $u_{i,t}$: industry-level idiosyncratic log-vol, shared AR(1) parameters
#
# **Linear (Rao-Blackwellised) state**: $f_t \in \mathbb{R}^2$, latent return
# factors with diagonal $\text{VAR}(1)$ dynamics.
#
# ### Observation model (conditionally Gaussian)
#
# ```math
# y_t \mid f_t, x_t \;\sim\; \mathcal{N}\!\bigl(\Lambda f_t,\;
#   \mathrm{diag}(\exp(a + g_t \mathbf{1} + u_t))\bigr)
# ```
#
# The per-series log-vol $h_{i,t} = a_i + g_t + u_{i,t}$ enters through an $\exp$
# link, making the system nonlinear. Conditional on $x_{0:T}$, however, the model
# in $(f_t, y_t)$ is linear-Gaussian, enabling Kalman filtering.
#
# The loadings $\Lambda \in \mathbb{R}^{5 \times 2}$ use a lower-triangular
# identification constraint (first two diagonal entries fixed to 1) to remove
# rotational indeterminacy.
#
# ### Volatility dynamics (outer / particle)
#
# ```math
# g_t = \phi_g\, g_{t-1} + \omega_t, \quad \omega_t \sim \mathcal{N}(0, \sigma_g^2)
# ```
# ```math
# u_{i,t} = \phi_u\, u_{i,t-1} + \zeta_{i,t}, \quad \zeta_{i,t} \sim \mathcal{N}(0, \sigma_u^2)
# ```
#
# with stationary initialisations $g_0 \sim \mathcal{N}(0, \sigma_g^2 / (1-\phi_g^2))$
# and similarly for $u_{i,0}$.
#
# ### Factor dynamics (inner / Rao-Blackwellised)
#
# ```math
# f_t = A f_{t-1} + \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q)
# ```
# with $A = \mathrm{diag}(\rho_1, \rho_2)$ and $Q = \mathrm{diag}(q_1^2, q_2^2)$.

# ## Implementation
#
# ### Outer (volatility) model

# Prior for the outer state $x_0 = [g_0, u_{1:5,0}]$, with stationary variances.
struct VolPrior{T<:Real} <: StatePrior
    φ_g::T
    σ_g::T
    φ_u::T
    σ_u::T
end

function SSMProblems.distribution(p::VolPrior{T}; kwargs...) where {T}
    σ²_g = p.σ_g^2 / (1 - p.φ_g^2)
    σ²_u = p.σ_u^2 / (1 - p.φ_u^2)
    μ₀ = @SVector zeros(T, 6)
    vars = SVector{6,T}(σ²_g, σ²_u, σ²_u, σ²_u, σ²_u, σ²_u)
    return MvNormal(μ₀, PDiagMat(vars))
end

# Transition density for the outer state.
struct VolDynamics{T<:Real} <: LatentDynamics
    φ_g::T
    σ_g::T
    φ_u::T
    σ_u::T
end

function SSMProblems.distribution(
    d::VolDynamics{T}, ::Integer, state::AbstractVector; kwargs...
) where {T}
    g, u₁, u₂, u₃, u₄, u₅ = state
    μ = SVector{6,T}(d.φ_g * g, d.φ_u * u₁, d.φ_u * u₂, d.φ_u * u₃, d.φ_u * u₄, d.φ_u * u₅)
    vars = SVector{6,T}(d.σ_g^2, d.σ_u^2, d.σ_u^2, d.σ_u^2, d.σ_u^2, d.σ_u^2)
    return MvNormal(μ, PDiagMat(vars))
end

# ### Inner (factor) model

# Diffuse prior for the initial factor state.
struct FactorPrior <: GaussianPrior end

GF.calc_μ0(::FactorPrior; kwargs...) = @SVector zeros(2)
# GF.calc_Σ0(::FactorPrior; kwargs...) = PDMat(SMatrix{2,2,Float64}(10.0 * I))
GF.calc_Σ0(::FactorPrior; kwargs...) = _pdmat(SMatrix{2,2,Float64}(10.0 * I))

# Diagonal VAR(1) factor dynamics.
struct FactorDynamics{T<:Real} <: LinearGaussianLatentDynamics
    ρ::SVector{2,T}
    q::SVector{2,T}
end

function GF.calc_A(d::FactorDynamics{T}, ::Integer; kwargs...) where {T}
    return SMatrix{2,2,T,4}(d.ρ[1], zero(T), zero(T), d.ρ[2])
end

GF.calc_b(::FactorDynamics{T}, ::Integer; kwargs...) where {T} = @SVector zeros(T, 2)

function GF.calc_Q(d::FactorDynamics{T}, ::Integer; kwargs...) where {T}
    return _pdmat(SMatrix{2,2,T,4}(d.q[1]^2, zero(T), zero(T), d.q[2]^2))
end

# Observation process: $y_t = \Lambda f_t + \varepsilon_t$ with time-varying
# diagonal noise covariance driven by the current outer (volatility) state.
struct FactorObservation{T<:Real} <: LinearGaussianObservationProcess
    Λ::SMatrix{5,2,T,10}  # lower-triangular loadings
    a::SVector{5,T}        # per-industry log-vol levels
end

GF.calc_H(obs::FactorObservation, ::Integer; kwargs...) = obs.Λ
GF.calc_c(::FactorObservation{T}, ::Integer; kwargs...) where {T} = @SVector zeros(T, 5)

# Hard upper bound on log-volatility fed into exp(). exp(20) ≈ 5e8, which is already
# far beyond any plausible monthly-return variance; this prevents particle drift from
# producing an ill-conditioned innovation covariance S = HΣH' + R.
const LOG_VOL_MAX = 20.0

# PDiagMat constructor with rrule so the SMatrix{5,5} cotangent from the KF rrule
# propagates correctly back through the diagonal-exponential parameterisation.
# A small jitter (1e-6) floors R to keep S = HΣH' + R positive-definite when extreme
# NUTS proposals drive exp(h) toward zero (H is 5×2, so HΣH' is rank-deficient alone).
function _make_obs_cov(h::SVector{N,T}) where {N,T}
    h_clamped = min.(h, T(LOG_VOL_MAX))
    return PDiagMat(exp.(h_clamped) .+ T(1e-6))
end

function ChainRulesCore.rrule(::typeof(_make_obs_cov), h::SVector{N,T}) where {N,T}
    h_clamped = min.(h, T(LOG_VOL_MAX))
    evh = exp.(h_clamped)
    R = PDiagMat(evh .+ T(1e-6))
    function _make_obs_cov_pullback(∂R)
        d = ChainRulesCore.unthunk(∂R)
        # Zero gradient for clamped components (clamp is not differentiable at the boundary,
        # but for stability we treat it as a hard stop).
        ∂h = SVector{N,T}(
            ntuple(i -> h[i] < T(LOG_VOL_MAX) ? d[i, i] * evh[i] : zero(T), N)
        )
        return (NoTangent(), ∂h)
    end
    return R, _make_obs_cov_pullback
end

# Transparent PDMat wrapper with rrule to bridge the SMatrix cotangent from the KF rrule.
_pdmat(A::AM) where {AM<:AbstractMatrix} = PDMat(A)

function ChainRulesCore.rrule(::typeof(_pdmat), A::AM) where {AM<:AbstractMatrix}
    P = _pdmat(A)
    pullback(∂P) = (NoTangent(), ChainRulesCore.unthunk(∂P))
    return P, pullback
end

function GF.calc_R(obs::FactorObservation{T}, ::Integer; new_outer, kwargs...) where {T}
    g, u₁, u₂, u₃, u₄, u₅ = new_outer
    h = obs.a .+ g .+ SVector{5,T}(u₁, u₂, u₃, u₄, u₅)
    return _make_obs_cov(h)
end

# ### Model constructor
#
# Assembles the `HierarchicalSSM` from raw (unconstrained) parameter vectors.
# Stability constraints are enforced via $\tanh$ (AR coefficients) and $\exp$
# (positive scales). The loadings use a lower-triangular identification:
# $\Lambda_{1,1} = \Lambda_{2,2} = 1$, $\Lambda_{1,2} = 0$; the 7 remaining
# entries are free.

function build_dfsv(λ_free, ρ_raw, log_q, atanh_φ_g, log_σ_g, atanh_φ_u, log_σ_u, a)
    T = promote_type(eltype(λ_free), typeof(atanh_φ_g))

    ρ = tanh.(ρ_raw)
    q = exp.(log_q)
    φ_g = tanh(atanh_φ_g)
    σ_g = exp(log_σ_g)
    φ_u = tanh(atanh_φ_u)
    σ_u = exp(log_σ_u)

    # Column-major layout: Λ[:,1] = [1, λ[1..4]], Λ[:,2] = [0, 1, λ[5..7]]
    Λ = SMatrix{5,2,T}(
        one(T),
        λ_free[1],
        λ_free[2],
        λ_free[3],
        λ_free[4],
        zero(T),
        one(T),
        λ_free[5],
        λ_free[6],
        λ_free[7],
    )

    outer_prior = VolPrior(φ_g, σ_g, φ_u, σ_u)
    outer_dyn = VolDynamics(φ_g, σ_g, φ_u, σ_u)

    factor_dyn = FactorDynamics(SVector{2,T}(ρ[1], ρ[2]), SVector{2,T}(q[1], q[2]))
    factor_obs = FactorObservation(Λ, SVector{5,T}(a))

    return HierarchicalSSM(outer_prior, outer_dyn, FactorPrior(), factor_dyn, factor_obs)
end;

# ## Data
#
# We use monthly value-weighted returns from the Ken French Data Library for the
# 5-industry classification (Consumer, Manufacturing, High-Tech, Healthcare, Other),
# covering January 1985 – December 2024 (480 months).

df = load_industry_data(; date_from=198501, date_to=202412)
dates = _yyyymm_to_date.(df.date)
Y = Matrix{Float64}(df[:, INDUSTRIES])

# Demean each series to remove the constant return component.
Y .-= mean(Y; dims=1)

plot_returns(df)

# Convert to a vector of observation vectors for the filter.
ys = [Vector{Float64}(Y[t, :]) for t in 1:size(Y, 1)];

# ## Inference
#
# ### Model specification
#
# The 20-dimensional parameter block for NUTS consists of:
# - 7 free loadings ($\lambda_{21}, \lambda_{31..51}, \lambda_{32..52}$)
# - 2 factor AR coefficients (raw, via $\tanh$)
# - 2 factor innovation log-scales
# - Common vol AR and log-scale ($\phi_g, \sigma_g$, raw)
# - Shared idiosyncratic vol AR and log-scale ($\phi_u, \sigma_u$, raw)
# - 5 per-industry log-vol levels $a_i$
#
# All AR parameters use the reparameterisation $\phi = \tanh(\tilde\phi)$ with a
# $\mathcal{N}(1.5, 0.7^2)$ prior on $\tilde\phi$, centering the prior on
# persistent ($\phi \approx 0.9$) processes as is typical for stochastic volatility.

@model function dfsv(ys)
    λ_free ~ MvNormal(zeros(7), I)
    ρ_raw ~ MvNormal(zeros(2), I)
    log_q ~ MvNormal(zeros(2), 0.5^2 * I)
    atanh_φ_g ~ Normal(1.5, 0.7)
    log_σ_g ~ Normal(log(0.15), 0.7)
    atanh_φ_u ~ Normal(1.5, 0.7)
    log_σ_u ~ Normal(log(0.15), 0.7)
    a ~ MvNormal(zeros(5), 2.0^2 * I)

    ssm = build_dfsv(λ_free, ρ_raw, log_q, atanh_φ_g, log_σ_g, atanh_φ_u, log_σ_u, a)
    return x ~ SSMTrajectory(ssm, KF(), ys)
end

# ### Sampler
#
# The particle Gibbs sampler alternates between:
# 1. A conditional SMC sweep over the volatility path $x_{0:T}$ using the
#    Rao-Blackwellised particle filter with **ancestor sampling** (PGAS), and
# 2. A NUTS step over the 20-dimensional parameter block, conditioned on the
#    current volatility path.
#
# Ancestor sampling (CSMCAS) significantly improves mixing of the reference
# trajectory at negligible extra cost. Zygote is used as the AD backend because
# the Kalman filter log-likelihood has an analytical reverse-mode rule that Zygote
# picks up via ChainRules.

rng = MersenneTwister(42)

N_particles = 1000
N_iter = 50
N_adapts = 10

model = dfsv(ys)

sampler = HMC(0.01, 10)
# sampler = NUTS(0.85)
pg = ParticleGibbs(
    CSMCAS(RBPF(BF(N_particles), KF())), sampler; adtype=ADTypes.AutoZygote()
)

chain = AbstractMCMC.sample(
    rng, model, pg, N_iter; n_adapts=N_adapts, progress=true, chain_type=MCMCChains.Chains
)

# ## Results

# ### Posterior parameter traces
#
# The traces (post-adaptation) illustrate NUTS mixing across the 20-dimensional
# parameter block. Slow mixing or high autocorrelation in random-walk MH on this
# block would be expected; NUTS largely avoids this by exploiting gradient
# information.

plot_chains(chain; burnin=N_adapts)

# ### Posterior volatility paths
#
# To recover the smoothed volatility paths we re-run a forward filter using the
# posterior mean parameters and collect the particle states via a callback.

post = MCMCChains.summarize(chain[(N_adapts + 1):end])

function posterior_mean_params(chain, burnin)
    get_mean(k) = mean(Array(chain[(burnin + 1):end, k, 1]))
    λ_free = [get_mean("λ_free[$i]") for i in 1:7]
    ρ_raw = [get_mean("ρ_raw[$i]") for i in 1:2]
    log_q = [get_mean("log_q[$i]") for i in 1:2]
    atanh_φ_g = get_mean(:atanh_φ_g)
    log_σ_g = get_mean(:log_σ_g)
    atanh_φ_u = get_mean(:atanh_φ_u)
    log_σ_u = get_mean(:log_σ_u)
    a = [get_mean("a[$i]") for i in 1:5]
    return λ_free, ρ_raw, log_q, atanh_φ_g, log_σ_g, atanh_φ_u, log_σ_u, a
end

params = posterior_mean_params(chain, N_adapts)
ssm_post = build_dfsv(params...)

cb = GF.AncestorCallback(nothing)
states, _ = GF.filter(rng, ssm_post, RBPF(BF(N_particles), KF()), ys; callback=cb)

# Collect particle outer states at each time step from the ancestry tree.
paths = GF.get_ancestry(cb.tree)
T_len = length(ys)
vol_paths = [[path[t].x for path in paths] for t in 1:T_len]

plot_volatilities(vol_paths, dates)
