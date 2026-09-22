# GeneralisedFilters.jl

Composable state-space filtering and smoothing, with Rao–Blackwellised particle Gibbs and
HMC parameter updates. The model interface comes from the neighbouring `SSMProblems/`
package, which owns the process types and generics; GeneralisedFilters adds parameter
atoms, conditioning and the algorithms, and integrates with Turing.jl.

## Conditional marginalisation

```julia
using GeneralisedFilters, StaticArrays, Distributions

function build(θ)
    Q = exp(θ[1]) * SMatrix{1,1}(1.0)
    inner_dyn((; t, x_prev, x_new)) = LinearGaussianDynamics(
        SMatrix{1,1}(0.8), SA[0.1x_prev + 0.2x_new], Q,
    )
    StateSpaceModel(
        DistributionPrior(Normal()),
        DistributionDynamics((t, x) -> Normal(0.9x, 0.3)),
        GaussianPrior(SA[0.0], SMatrix{1,1}(1.0)),
        inner_dyn,
        LinearGaussianObservation(SMatrix{1,1}(1.0), SA[0.0], SMatrix{1,1}(0.2)),
    )
end

xs = ReferenceTrajectory(0.1, [0.2, -0.3, 0.4])
ys = [SA[0.2], SA[-0.1], SA[0.3]]
inner = condition_inner(build([-1.0]), xs)
ll = marginal_loglikelihood(inner, KF(), ys)
objective(θ) = trajectory_logdensity(build(θ), KF(), xs, ys)
```

The conditional view shares the same evaluator as an ordinary analytical SSM. The complete
trajectory objective includes outer densities and marginalises inner states. Add parameter
priors once through the host inference system. ForwardDiff differentiates the generic primal;
loading Mooncake enables the static-array Kalman reverse rule.

For particle Gibbs, construct `ConditionalSMC(RBPF(BF(N), KF()), AncestorSampling())` and use
it through the standalone `ParticleGibbs` sampler or the Turing adapter. Conditional
SMC requires a resampler with a conditional law (multinomial, systematic or stratified);
exact RB backward methods require an unrepaired analytical filter.

## Documentation and migration

- [Models and conditioning](GeneralisedFilters/docs/src/models/linear-gaussian.md)
- [Particle Gibbs and Turing](GeneralisedFilters/docs/src/inference.md)
- [Migrating from 0.4.2](GeneralisedFilters/docs/src/migration.md)
- [Release notes](GeneralisedFilters/CHANGELOG.md)

Version 0.5 replaces the `calc_*`/keyword-conditioning interface with whole-component
closures, plain Gaussian states, and explicit conditional models. Reference trajectories
contain outer states only; rebuild conditional views after changing a trajectory or θ.

From the repository root, develop the shared dependency before running package tests:

```sh
julia --project=GeneralisedFilters -e 'using Pkg; Pkg.develop(path="SSMProblems"); Pkg.test()'
```

The package split requires SSMProblems 0.7 and Turing 0.47–0.49; see the migration
notes for compatibility and release order.
CPU tests include the AD and Turing integrations. GPU runtime tests require CUDA hardware.
