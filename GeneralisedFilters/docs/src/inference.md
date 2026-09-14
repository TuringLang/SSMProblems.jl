# Particle Gibbs and Turing

A Rao–Blackwellised particle filter stores each particle's sampled outer state and its inner
filtering distribution. A proposal receives that complete `RBState`, and proposes only the
outer component. Conditional SMC persists only the outer trajectory and recomputes inner
beliefs under the current parameters.

```julia
using GeneralisedFilters, AdvancedHMC, AbstractMCMC, ADTypes, Mooncake
pf = RBPF(BF(100), KF())
csmc = ConditionalSMC(pf, AncestorSampling())
pg = ParticleGibbs(csmc, AdvancedHMC.NUTS(0.8); adtype=AutoMooncake())
model = ParticleGibbsModel(parameter_prior, ParameterisedSSM(build, observations))
chain = AbstractMCMC.sample(rng, model, pg, 1000)
```

Use `AutoForwardDiff()` and load ForwardDiff for forward-mode parameter updates. Parameters
passed to the standalone sampler are in the coordinates of `parameter_prior`; use a builder
with explicit transformations, or Turing for automatic constrained-parameter handling.

## Turing models

Within a Turing model, construct the SSM from the current parameter values and express the
trajectory through `SSMTrajectory(ssm, KF(), observations)`. Its density includes the outer
trajectory density and the inner marginal likelihood. Turing supplies parameter priors and
transformation Jacobians; do not add those again to the trajectory factor.

```julia
using Turing, GeneralisedFilters, AdvancedHMC, AbstractMCMC, MCMCChains, ADTypes, ForwardDiff
@model function inference_model(ys)
    q ~ LogNormal(-1.0, 0.5)
    ssm = build_from_variance(q)
    x ~ SSMTrajectory(ssm, KF(), ys)
end
# ParticleGibbs alternates the parameter NUTS and trajectory CSMC updates.
csmc = ConditionalSMC(RBPF(BF(100), KF()), AncestorSampling())
sampler = ParticleGibbs(csmc, AdvancedHMC.NUTS(0.8); adtype=AutoForwardDiff())
chain = AbstractMCMC.sample(rng, inference_model(ys), sampler, 1000;
    n_adapts=200, progress=false, chain_type=MCMCChains.Chains)
```

See the executable examples in `examples/PGAS Example`. Parameters may enter both outer and
inner processes, including the conditional inner prior. Each trajectory update rebuilds the
conditional target; cached HMC energy and gradients must be refreshed before another update.
AD preparation is rebuilt at the integration boundary for correctness.

The Turing trajectory representation uses fixed-dimensional real vectors. Variable-length
outer states require a different storage/integration contract. Supporting a conditional
likelihood does not make arbitrary Turing programs analytically marginalisable.

The Turing adapter currently supports exactly one `SSMTrajectory` variable per model and
starts from prior parameter draws; `initial_params` is rejected explicitly. It is used as the
outer sampler shown above, not as a component of `Turing.Gibbs`. The standalone parameter
sampler supports `initial_params`.

## Stable Gaussian refreshment

`KF()` and `SRKF()` use `SqrtBackwardInformationPredictor()` by default for RB ancestor
sampling and backward simulation. For square-root arithmetic in the forward pass too,
use the same `SRKF()` in the RBPF and in `SSMTrajectory`:

```julia
pf = AuxiliaryParticleFilter(RBPF(BF(100; threshold=0.8), SRKF()), MeanPredictive())
csmc = ConditionalSMC(pf, BackwardSimulation())
# Inside the Turing model: x ~ SSMTrajectory(ssm, SRKF(), observations)
```

Supply `CovarianceFactor(F)` as a Gaussian atom's covariance when a prior or process
covariance is naturally available as `F * F'`, including rectangular/rank-deficient factors.
The square-root route consumes those factors directly. Observation noise must remain
positive definite. Explicit noise regularization belongs in the model and must be shared
by trajectory and parameter updates; covariance clipping inside the forward filter is
not compatible with the analytical backward formulas.

ForwardDiff and Mooncake differentiate the conditional forward likelihood; HMC does not
need derivatives through particle selection or backward messages. Dense reverse-mode QR
requires full column rank whenever its output carries a nonzero derivative. Rank-deficient filtering support does not
imply differentiability through rank changes; use a smooth, fixed-rank parameterization
and validate its gradients. The square-root implementation introduces no eigenvalue floor.

`AncestorSampling()` respects the ESS trigger and currently updates the reference ancestor
only when ordinary resampling occurs. It preserves ancestry and accumulated weights on
skipped steps. `BackwardSimulation()` performs its backward pass after the adaptive forward
sweep. APF lookahead weights guide ordinary ancestor selection without changing the target
backward weights.

## Possible MH correction of approximate backward weights

A future fallback could use approximate backward scores to propose an ancestor and apply
an MH correction with the target scores of only the current and proposed ancestors. With
a fixed number of proposals per step, suffix evaluation would cost `O(T^2)` in total,
alongside `O(NT)` particle/proposal work. This is the MH-within-PGAS construction in
[Lindsten et al., §6.1](https://jmlr.org/papers/volume15/lindsten14a/lindsten14a.pdf).
It is not currently an implemented refreshment strategy.

One global MH correction after an arbitrary approximate particle sweep is not automatically
valid: it requires the reverse proposal law, or a proven reversible proposal kernel for an
evaluable surrogate target. All target evaluations, including HMC, must remain consistent.
An MH correction still requires a stable target likelihood evaluator.
