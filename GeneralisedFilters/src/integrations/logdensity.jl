using LogDensityProblems: LogDensityProblems
import Distributions: logpdf
export ParameterisedSSM, SSMParameterLogDensity

## PARAMETERISED SSM ###########################################################################

"""
    ParameterisedSSM(build, observations)

A parameterised state-space model that maps parameter vectors to concrete SSMs.

# Fields
- `build`: A callable `θ -> StateSpaceModel` that constructs an SSM from parameters.
  Fixed model components should be captured via closure.
- `observations`: The observation sequence y₁:T.

# Example
```julia
function build_model(θ, fixed)
    b = θ[1:2]
    dyn = LinearGaussianDynamics(fixed.A, b, fixed.Q)
    return StateSpaceModel(fixed.prior, dyn, fixed.obs)
end

pssm = ParameterisedSSM(θ -> build_model(θ, fixed), observations)
model = pssm.build(θ)  # returns a concrete SSM
```
"""
struct ParameterisedSSM{F,YT}
    build::F
    observations::YT
end

## SSM PARAMETER LOG-DENSITY ###################################################################

"""
    SSMParameterLogDensity(prior, param_model, af, trajectory)
    SSMParameterLogDensity(prior, param_model, trajectory)

Log-density for SSM parameters θ conditioned on a fixed trajectory:

    log p(θ | trajectory, y) ∝ log p(θ) + log p(trajectory, y | θ)

Implements the `LogDensityProblems` interface.

# Fields
- `prior`: Multivariate prior distribution on the parameter vector θ
- `param_model`: A `ParameterisedSSM` mapping θ to an SSM
- `af`: Inner analytical filter for HierarchicalSSM (e.g., `KalmanFilter()`), or `nothing`
  for regular SSMs
- `trajectory`: Current reference trajectory ([`ReferenceTrajectory`](@ref) indexed from 0)
"""
struct SSMParameterLogDensity{PT,MT<:ParameterisedSSM,AFT,TT}
    prior::PT
    param_model::MT
    af::AFT
    trajectory::TT
end

function SSMParameterLogDensity(prior, param_model::ParameterisedSSM, trajectory)
    return SSMParameterLogDensity(prior, param_model, nothing, trajectory)
end

function LogDensityProblems.capabilities(::Type{<:SSMParameterLogDensity})
    return LogDensityProblems.LogDensityOrder{0}()
end

function LogDensityProblems.dimension(ld::SSMParameterLogDensity)
    return length(ld.prior)
end

function LogDensityProblems.logdensity(ld::SSMParameterLogDensity{<:Any,<:Any,Nothing}, θ)
    model = ld.param_model.build(θ)
    return logpdf(ld.prior, θ) +
           trajectory_logdensity(model, ld.trajectory, ld.param_model.observations)
end

function LogDensityProblems.logdensity(
    ld::SSMParameterLogDensity{<:Any,<:Any,<:AbstractFilter}, θ
)
    model = ld.param_model.build(θ)
    return logpdf(ld.prior, θ) +
           trajectory_logdensity(model, ld.af, ld.trajectory, ld.param_model.observations)
end
