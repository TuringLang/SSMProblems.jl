# Particle Gibbs and Turing

When both model parameters and latent states are unknown, we want samples from their joint
posterior given the observations. Particle Gibbs alternates two updates: it samples a
state trajectory with the parameters held fixed, then samples parameters with that
trajectory held fixed.

For a Rao-Blackwellised model, only the outer trajectory is sampled. The inner states are
integrated out by an analytical filter in both updates. In the changing-volatility example
from [Models and conditioning](models/linear-gaussian.md), this means sampling the log
variance over time while integrating out the underlying Gaussian signal.

The parameter update can use Hamiltonian Monte Carlo (HMC), including the No-U-Turn Sampler
(NUTS). Its gradients pass through the conditional likelihood computed by the analytical
filter. They do not pass through particle resampling. GeneralisedFilters supports
ForwardDiff for forward-mode differentiation and Mooncake for reverse-mode differentiation.
The latter can be useful when there are many parameters.

## Define a model in Turing

Turing lets you specify parameter priors and handles transformations for constrained
parameters. The following example learns the variance `q` governing a latent log-variance
process. Conditional on that process, the signal and observations are linear-Gaussian.

```julia
using GeneralisedFilters, StaticArrays, Distributions, Random
using Turing, AdvancedHMC, AbstractMCMC, MCMCChains, ADTypes, ForwardDiff

function build_from_variance(q)
    inner_dyn((; t, x_prev, x_new)) = LinearGaussianDynamics(
        SMatrix{1,1}(0.8), SA[0.0], SMatrix{1,1}(exp(x_new)),
    )
    return StateSpaceModel(
        DistributionPrior(Normal()),
        DistributionDynamics((t, x) -> Normal(0.9x, sqrt(q))),
        GaussianPrior(SA[0.0], SMatrix{1,1}(1.0)),
        inner_dyn,
        LinearGaussianObservation(SMatrix{1,1}(1.0), SA[0.0], SMatrix{1,1}(0.2)),
    )
end

@model function inference_model(ys)
    q ~ LogNormal(-1.0, 0.5)
    ssm = build_from_variance(q)
    x ~ SSMTrajectory(ssm, KF(), ys)
end
```

`SSMTrajectory(ssm, KF(), ys)` connects the state-space model to Turing. Its density includes
both the outer trajectory density and the likelihood of `ys` after integrating out the
inner states. Include this factor once. Adding a separate likelihood for the same
observations would count them twice. Turing adds the parameter priors and transformation
Jacobians.

## Choose the trajectory and parameter samplers

The trajectory update uses conditional sequential Monte Carlo (CSMC), a particle filter
that retains the current trajectory as a reference. Ancestor sampling lets that reference
trajectory reconnect to other particle histories, which can improve mixing.

```julia
rng = Xoshiro(42)
ys = [SA[0.2], SA[-0.1], SA[0.3]]

pf = RBPF(BF(100), KF())
csmc = ConditionalSMC(pf, AncestorSampling())
sampler = ParticleGibbs(csmc, AdvancedHMC.NUTS(0.8); adtype=AutoForwardDiff())
chain = AbstractMCMC.sample(rng, inference_model(ys), sampler, 1000;
    n_adapts=200, progress=false, chain_type=MCMCChains.Chains)
```

`BF(100)` selects a bootstrap filter with 100 particles. `RBPF` combines it with a Kalman
filter for the inner state. `ParticleGibbs` alternates the CSMC trajectory update with NUTS
parameter updates, using a target acceptance probability of 0.8 in this example.

To use reverse-mode differentiation, load `Mooncake` and replace `AutoForwardDiff()` with
`AutoMooncake()`. Both choices differentiate the same model. Parameters can enter the outer
process, the inner process, and the inner initial-state prior. After each trajectory
update, the sampler refreshes the parameter objective and its AD preparation for the new
conditional distribution.

See the [particle Gibbs example](https://github.com/TuringLang/SSMProblems.jl/blob/main/GeneralisedFilters/examples/PGAS%20Example/rb_ssm.jl)
for a longer inference script.

### Turing integration limits

The current adapter supports one `SSMTrajectory` variable per Turing model. The sampled
trajectory must have continuous real-valued states with a dimension that stays fixed across
time and parameters. Discrete or variable-dimensional outer states are not supported by
this adapter, even though standalone CSMC is not restricted to this representation.

Use `ParticleGibbs` as the sampler passed to `sample`, as above. It cannot currently be used
as a component of `Turing.Gibbs`. The adapter initialises parameters from their priors and
rejects the `initial_params` keyword. You must also supply the analytically tractable inner
model explicitly. The adapter does not discover which parts of an arbitrary Turing model
can be integrated out.

## Particle Gibbs without Turing

The standalone interface takes a parameter prior and a function that builds the
state-space model. It uses the same trajectory and parameter samplers. For the model above,
one option is to sample the log variance and transform it in the builder:

```julia
parameter_prior = MvNormal([-1.0], [0.5^2;;])
build(θ) = build_from_variance(exp(θ[1]))
parameterised = ParameterisedSSM(build, ys)
model = ParticleGibbsModel(parameter_prior, parameterised)
chain = AbstractMCMC.sample(rng, model, sampler, 1000; n_adapts=200)
```

Here the prior is on the log variance itself. The standalone sampler operates in the
coordinates of `parameter_prior` and does not automatically transform constrained
parameters. Choose suitable unconstrained coordinates and include any required density
adjustment when expressing a prior in different coordinates. Unlike the Turing adapter,
the standalone sampler accepts `initial_params`.

## Resampling and trajectory updates

Each Rao-Blackwellised particle stores an outer state and an inner filtering distribution
in an `RBState`. Custom proposals receive both but propose only an outer state. CSMC retains
only the outer reference trajectory, then recomputes its inner distributions under the
current parameters.

Ordinary particle resampling and ancestor sampling are distinct operations. In the current
implementation, `AncestorSampling()` updates the reference ancestor only on steps where
the effective sample size (ESS) criterion triggers ordinary resampling. On skipped steps,
ancestry and accumulated weights are preserved. `BackwardSimulation()` instead samples a
trajectory in a backward pass after the forward particle sweep.

An auxiliary particle filter can use a prediction of the next observation to guide
resampling. For example:

```julia
pf = AuxiliaryParticleFilter(RBPF(BF(100; threshold=0.8), SRKF()), MeanPredictive())
csmc = ConditionalSMC(pf, BackwardSimulation())
```

The lookahead weights guide ordinary ancestor selection. The backward weights still use
the target model. This example also selects the square-root Kalman filter, `SRKF()`, for
the inner states. If you use it with Turing, select `SRKF()` in `SSMTrajectory` too.

## Numerical stability

Gaussian filtering can lose precision when covariance matrices are poorly conditioned.
`SRKF()` performs the forward calculations with covariance factors. Both `KF()` and
`SRKF()` use `SqrtBackwardInformationPredictor()` by default for Rao-Blackwellised ancestor
sampling and backward simulation.

If a prior or process covariance is naturally available as `F * F'`, pass
`CovarianceFactor(F)` as the covariance of its Gaussian component. The square-root filter
uses the factor directly, including rectangular factors that represent rank-deficient
covariances. Observation noise must remain positive definite. The square-root
implementation does not add an eigenvalue floor.

If your model needs additional noise for numerical stability, include that noise in the
model used by both the trajectory and parameter updates. Clipping covariances inside the
forward filter is not compatible with the analytical backward formulas.

Filtering with a rank-deficient covariance does not guarantee that gradients exist through
rank changes. Dense reverse-mode differentiation of QR requires full column rank when its
output carries a nonzero derivative. Use a smooth parameterisation with fixed rank and
check its gradients before using it for HMC.
