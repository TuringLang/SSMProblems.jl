# GeneralisedFilters

GeneralisedFilters implements analytical and particle filtering, smoothing, and particle
Gibbs for state-space models. A hierarchical model can marginalise its inner Gaussian
states while sampling only its outer trajectory. Parameter updates differentiate the
conditional marginal likelihood using ForwardDiff or Mooncake.

Install with `import Pkg; Pkg.add("GeneralisedFilters")`. Load `Mooncake` to enable the
handwritten reverse rule, or `Turing` for the Gibbs integration. These packages are optional.
The model interface comes from SSMProblems, whose process types and generics are
re-exported here, so importing it separately is not required.

Small, fixed-dimensional states can use StaticArrays. Ordinary arrays remain supported;
the optimised reverse rule applies to immutable floating-point static arrays. Model builders
must preserve the numeric types supplied by AD.

See [Models and conditioning](models/linear-gaussian.md),
[Particle Gibbs and Turing](inference.md), and [Migrating to 0.6](migration.md).
