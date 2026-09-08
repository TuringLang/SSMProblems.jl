# Particle Gibbs and Turing

A Rao–Blackwellised particle filter stores each particle's sampled outer state and its inner
filtering distribution. A proposal receives that complete `RBState`, and proposes only the
outer component. Conditional SMC persists only the outer trajectory and recomputes inner
beliefs under the current parameters.

```julia
using GeneralisedFilters, AdvancedHMC, AbstractMCMC, ADTypes, Mooncake
pf = RBPF(BF(100; resampler=GeneralisedFilters.Multinomial()), KF())
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
csmc = ConditionalSMC(RBPF(BF(100; resampler=GeneralisedFilters.Multinomial()), KF()), AncestorSampling())
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
