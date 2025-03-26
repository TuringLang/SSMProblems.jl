using GeneralisedFilters
using SSMProblems
using LinearAlgebra
using MatrixEquations

include("linear_dsge.jl")

#=
# RANK Model

Suppose we observe 3 variables: $y_{t}$ deviation from domestic output, $\pi_{t}$ rate of
inflation, and $i_{t}$ the interest rate.
$$
y_{t} = E\left[y_{t+1}\right] - \frac{1}{\gamma} (i_{t} - E\left[\pi_{t+1}\right]) + \omega^{d}_{t}
$$
is the log form of the Euler equation (in terms of goods produced $y_{t})$ for intertemporal
choice with risk aversion parameter $\gamma$.
$$
\pi_{t} = \beta E\left[\pi_{t+1}\right] + \kappa y_{t} - \omega^{s}_{t}
$$
represents the new Keynesian Phillips curve given discount factor $\beta$ and the slope of
the NKPC $\kappa = \frac{(1-\theta)(1 - \theta \beta)}{\theta} (\gamma + \varphi)$, which
itself is an expression in terms of reactive firms $\theta$ and labor preferences $\varphi$.
$$
i_{t} = \phi^{i} i_{t-1} + (1-\phi^{i}) (\phi^{\pi} \pi_{t} + \phi^{y} y_{t}) + \omega^{m}_{t}
$$
defines the monetary policy rule with persistence $\phi^{i}$ and coefficients $\phi^{\pi}$
and $\phi^{y}$.

**New Keynesian** models allow for a property called *sticky prices* in which case only a
fraction of the goods producing firms properly react to changes in inflation.

More information can be found in ([Leeper & Leith, 2016](https://www.sciencedirect.com/science/article/pii/S1574004816000136))
who explain the basic setup in section 3.1, but this specific variation of the model comes
from ([Wolf, 2020](https://www.aeaweb.org/articles?id=10.1257/mac.20180328))

### Why is this useful?
 - aren't meant for forecasting since aggregate trends are rarely reflective of heterogeneous economies (this raises a new class of HANK models).
 - good for capturing narrative elements to structural models
 - allows us to perform *structural identification* using impulse response analysis

# Shocks

terms $\omega^{z}_{t}$ represent shocks to demand $d$, supply $s$, and interest rates $m$. In my version of the model, these are modeled as endogenous $AR(1)$ processes with persistence $\rho_{z}$ and shock variance $\sigma_{z}^2$
$$
\begin{align*}
    \omega^{d}_{t} &= \rho^{d} \omega^{d}_{t-1} + \sigma^{d} \varepsilon^{d}_{t} \\
    \omega^{s}_{t} &= \rho^{s} \omega^{s}_{t-1} + \sigma^{s} \varepsilon^{s}_{t} \\
    \omega^{m}_{t} &= \rho^{m} \omega^{m}_{t-1} + \sigma^{m} \varepsilon^{m}_{t}
\end{align*}
$$

we have the freedom to pick any kind of behavior, this is really more of an art than a science.
=#

function new_keynesian_model(
    β::T, γ::T, φ::T, θ::T, ϕπ::T, ϕy::T, ϕi::T, ρd::T, ρs::T, ρm::T, σd::T, σs::T, σm::T
) where {T<:Real}
    Γ0 = zeros(T, (8, 8))
    Γ1 = zeros(T, (8, 8))
    Ψ = zeros(T, (8, 3))
    Π = zeros(T, (8, 2))
    C = zeros(T, 8)

    # endogenous model equations
    Γ0[1, :] = [1 0 (1 / γ) -1 -(1 / γ) -1 0 0]
    Γ0[2, :] = [-(1 - θ) * (1 - θ * β) / θ * (γ + φ) 1 0 0 -β 0 1 0]
    Γ0[3, :] = [-(1 - ϕi) * ϕy -(1 - ϕi) * ϕπ 1 0 0 0 0 -1]

    Γ1[3, 3] = ϕi

    # forward lookers
    Γ0[4, 1] = one(T)
    Γ0[5, 2] = one(T)

    Γ1[4, 4] = one(T)
    Γ1[5, 5] = one(T)

    Π[4:5, :] = I(2)

    # shock processes
    Γ0[6:end, 6:end] = I(3)
    Γ1[6:end, 6:end] = diagm([ρd, ρs, ρm])
    Ψ[6:end, :] = diagm([σd, σs, σm])

    return LinearRationalExpectation(Γ0, Γ1, Ψ, Π, C)
end

# create the DSGE model
dsge = new_keynesian_model(
    0.995, 1.0, 1.0, 0.75, 1.5, 0.1, 0.9, 0.8, 0.9, 0.2, 1.6013, 0.9488, 0.2290
)

# generate the state space form
model = StateSpaceModel(dsge)
