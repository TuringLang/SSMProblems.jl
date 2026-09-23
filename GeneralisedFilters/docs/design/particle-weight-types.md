# Particle weight types before 0.5

Status: implemented, 2026-09-23. The agreed nonnumeric markers,
`add_logweight`, first-step promotion boundary and CSMC history precision are implemented. The intended policy is to defer the weight
scalar type until an actual density is evaluated, without introducing a general-purpose
symbolic numeric system.

## Findings

A latent state need not have the same scalar type as its log density. Integer states can
have floating-point likelihoods, and parameters can introduce AD scalars even when states
are ordinary numbers. A mandatory weight type inferred from the initial state is therefore
not a suitable default.

`TypelessZero` currently serves two legitimate purposes: initial unweighted particles and
an exactly zero bootstrap proposal correction. The latter avoids computing identical
transition and proposal log densities. Both uses should be preserved.

The current `Number` interface has concrete defects:

- `convert(DualType, TypelessZero())` is ambiguous with ForwardDiff's conversion methods.
- `convert(BigFloat, TypelessBaseline(N))` evaluates `log(N)` in Float64 first. At 256-bit
  precision, N=3 gives an error of approximately 9.07e-17 relative to `log(BigFloat(3))`.
- `iszero(TypelessBaseline(1))` returns false although the represented value is log(1).
- The uniform-weight `softmax` method chooses Float64 before any weight type is known.

The first three behaviours were reproduced against this branch. These are independent
of the Kalman accumulator, which now starts from its first likelihood increment.

The impact extends beyond initialisation. Symbolic values appear in ordinary and RB
particle prediction, weight normalisation and auxiliary-filter baseline corrections.
`TypelessBaseline` is also constructed during APF resampling after weights are numeric.
A change limited to the first particle update would miss this use.

## Alternatives

1. Add ForwardDiff-specific conversions and correct baseline rounding. This is a small
   compatibility repair, but retains broad numeric promotion and invites further
   ambiguities with other scalar types. It does not establish a clear sentinel boundary.
2. Require a weight scalar type when constructing every filter. This makes storage explicit,
   but couples the algorithm to each model's density and AD backend. It is unnecessarily
   restrictive as a default.
3. Keep deferred initialisation with internal markers and explicit weight operations.
   This preserves model flexibility while avoiding generic numeric conversion of markers.

Recommend option 3. An optional explicit storage type can be considered separately if a
specific GPU or preallocation use case requires it.

## Proposed contract

- A successful observation update produces concrete numeric log weights and a numeric
  likelihood increment. Their types follow the density calculations and ordinary promotion.
- Before the first numeric contribution, weights may be internally marked as unweighted.
  A bootstrap proposal may similarly report an exact zero correction without choosing a
  floating-point type.
- These markers are not numbers. They do not participate in arbitrary `convert`, promotion,
  `logsumexp`, `softmax` or external arithmetic.
- Internal weight addition returns the numeric operand unchanged when the other operand
  is the zero marker. Two markers produce a marker. Numeric operands use ordinary addition.
- Equal-weight normalisation retains N as an integer until a numeric log sum is available.
  It then computes log(N) in that calculation's scalar type, converting N before taking
  the logarithm. For AD scalars this count contributes zero parameter sensitivity.
- APF baseline formulas and proposal corrections retain their existing mathematical
  meaning. Only their representation and explicit resolution change.
- Uniform resampling before the first density needs its own explicit probability-storage
  policy. It must not accidentally establish the later likelihood accumulator's type.
  Preserve the existing default behaviour initially and test device-specific paths.
- Once numeric weights are available, history storage uses their actual type. Later type
  changes must be promoted deliberately or rejected at a documented storage boundary.

A small standalone prototype with non-Number markers, explicit addition and
`log(oftype(logsum, N))` passed Float32, Float64 and 256-bit BigFloat checks, including
ForwardDiff derivatives. This establishes feasibility of the basic operations only.
It is not validation of a full particle filter, Mooncake, APF or GPU implementation.
For very large N, converting N to a low-precision scalar can round or overflow. The chosen
normalisation helper needs an explicit supported range or a numerically justified fallback.
BigFloat evaluation also follows Julia's active precision context.

## Compatibility and implementation scope

`TypelessZero` and `TypelessBaseline` are not exported, but particle fields and intermediate
results expose them to custom code. Treat changes to that representation as a migration
concern rather than assuming private names make the change harmless.

Keep ordinary user construction unchanged: `BF`, `PF`, `RBPF`, `Particle(state, ancestor)`
and model builders should not require a new weight-type argument. Preserve numeric
corrections returned by custom proposals. Internal zero-marker corrections remain a
special case. Document what custom particle-level overrides can receive before the first
observation update and provide a supported weight-combination operation if those overrides
must handle unweighted particles. Choose that operation's name and scope during the patch,
before documenting it as a stable extension API.

Implementation touches:

1. Particle constructors, the unweighted-particle alias, baseline representation and
   normalisation in `containers.jl`.
2. Standard and RB particle prediction/update, bootstrap corrections, and preservation of
   proposal corrections in `algorithms/particles.jl` and `algorithms/rbpf.jl`.
3. Ordinary/APF resampling, initial uniform weights, ancestor weights and CSMC initial-state
   handling. Check all arithmetic involving `log_weight`, not just explicit marker names.
4. History and GPU consumers. `DenseParticleContainer`'s convenience constructors currently
   choose Float64 weights, while its first-observation constructor infers the actual type.
   CSMC uses that constructor but explicitly converts weights with `Float64.(...)` both
   on creation and append. It therefore does not currently preserve inferred precision.
   Decide whether to remove those conversions and whether convenience constructors need
   an explicit weight-type option before freezing the 0.5 interface.

Use a function boundary between initial unweighted processing and the numeric time loop
where needed for inference. Do not allocate unions of symbolic and numeric weights in the
steady-state particle array. Do not evaluate a random proposal twice to discover its type.

## Release validation

Before accepting an implementation:

- Check Float32/Float64 likelihoods and high-precision normalisation, including N=1 and
  non-power-of-two particle counts. Check state and density types that differ.
- Compare standard PF, guided PF, RBPF and APF likelihoods and normalised weights against
  the current formulas on fixed particles. Include adaptive skipped-resampling steps,
  conditional resampling, ancestor sampling and backward simulation.
- Reproduce and close the conversion path from issue #153. Validate local numeric AD
  through weight updates and reductions with ForwardDiff and Mooncake. Nested ForwardDiff
  and reused reverse-mode preparations need explicit tests.
- Exercise the full PF execution path with AD inputs where supported, without claiming
  that differentiation through discrete resampling gives an exact likelihood gradient.
  RBPG parameter gradients still use the conditional analytical likelihood.
- Test particle-Gibbs/Turing integration, custom proposal corrections, and GPU resampling.
- Check inference and allocations before and after the first update, using existing
  benchmarks. Keep the first-update type transition outside the repeated numeric loop.

Do not claim this design fixes every precision choice in the package. Probability storage,
random-number generation, state arithmetic and likelihood accumulation are distinct
choices. This work should make those choices explicit where the weight interface needs
them, without adding a global Float64 policy or promising arbitrary numeric-type support.

## Detailed lifecycle and decisions for discussion

Follow-up audit, 2026-09-23. The proposal needs more than replacing a numeric subtype.

### Exact zero versus unknown value

The zero marker denotes a known additive identity, not missing data or an unknown weight.
Keep one such marker for unweighted particles and exact bootstrap corrections. Do not use
it to suppress a model density evaluation. An observation method returning numerical zero
still supplies a numeric type. A density callback should return a concrete real scalar,
including when its value is zero. Returning the marker from all density callbacks would
leave precision unresolved and is outside the proposed model contract.

Use explicit dispatch for zero/zero, zero/numeric and numeric/zero. Numeric/numeric
addition keeps Julia's promotion. A general method accepting arbitrary objects should
not silently bless invalid density results. The initial particle collection should be
uniformly marked, rather than an arbitrary mixture of markers and numbers.

### The first numeric contribution may precede the observation

Ordinary initial particles have zero log weights and zero baseline. Prediction carries a
pre-observation normaliser log(N). A guided proposal supplies a numeric correction before
the observation. APF lookahead produces numeric scores before the first resampling and
may produce numeric inverse-lookahead weights. Initial PGAS ancestor selection also
combines marked weights with numeric future densities.

Do not impose a rule that only the observation may establish the type. Use ordinary
promotion across the contributions actually evaluated, then establish stable storage
from the completed first update. Never repeat sampling or model evaluation to infer types.
A standalone prediction result may legitimately remain marked for a bootstrap filter.

APF requires explicit normaliser differences, including cancellation of equal-count
normalisers when both sides are symbolic. Preserve the current APF formula:

    baseline = -((LSE(aux + old) - LSE(old)) + (LSE(new) - log(N)))

Resolve symbolic terms against numeric terms where present. Handle same-count symbolic
cancellation exactly, without needing a floating-point type. Review reachable combinations
before defining helper methods. Avoid creating a general symbolic expression algebra.

### Public extension boundary

Two viable choices exist:

- Expose a small supported operation such as `add_logweight`, which works on initial
  marked weights and ordinary numeric weights. Custom particle-level overrides use it.
- Keep all marker-aware operations private and expose only higher-level extension hooks
  that never require callers to manipulate an initial log weight.

Recommend the first for 0.5 because the package already permits particle-level overrides.
Changing that extension model would be a larger redesign. Document that initial raw
`log_weight` values are bookkeeping values and cannot be passed to arbitrary numeric
libraries. `get_weights` should continue to provide usable sampling probabilities.
Renaming or retaining the private marker name is less important than this public contract.

### Resampling is a separate AD boundary

Current systematic and stratified resamplers call `rand(rng, WT)` where WT comes from the
weights. A dual weight type therefore reaches random-number generation as well as symbolic
arithmetic. Fixing zero conversion alone does not establish support for a full AD-valued
filter sweep. ESS comparisons and ancestor selection also depend on parameter values.

For this cleanup, validate AD through numeric weight updates, normalisation and suitable
proposal calculations with resampling disabled or externally fixed. Do not silently strip
derivatives in the sampler and advertise the result as a complete differentiable-filter
estimator. A future explicit resampling-gradient policy must define how primal probabilities,
random draws, selected ancestors and gradient estimators interact.

Before any numeric density exists, retain the current ordinary uniform probabilities for
resampling. Their Float64 representation must not determine subsequent log-weight precision.
A device-specific uniform allocation can be addressed through the resampling interface.

### History precision and changing types

Remove the explicit CSMC Float64 history conversions if preserving numeric weight types is
adopted as the 0.5 contract. This concerns posterior weight storage even when no AD is taken
through CSMC. It is separate from a promise to differentiate CSMC.

A completed first update is a useful storage boundary but cannot predict arbitrary later
type changes. For example, a later likelihood callback can introduce Float64 values or an
AD-active parameter into an initially Float32/inactive calculation. Initialisation inference
alone does not solve that problem.

Recommend stable numeric representations for the prepared repeated loop and stored history.
Builders should use consistent scalar representations across time, including zero partials
at times where an AD parameter is inactive. Do not add a broad rejection to generic code
that can already promote safely. Document and test constraints at actual fixed-storage or
AD-preparation boundaries. Automatic widening of every history and AD workspace is outside
this patch.

### Constant precision and degenerate weights

The count N has zero parameter derivative, but log(N) still requires rounding. Converting
N before log is suitable for ordinary Float32/Float64 particle counts and BigFloat, but can
overflow for Float16 even when log(N) is representable. A robust helper may evaluate that
constant in a wider type and round back, with an explicit rule for each supported scalar
family. This is distinct from widening all likelihood accumulation. Do not claim universal
numeric-type support based on the prototype.

All-negative-infinity weights are numeric, not unresolved. They cannot be normalised into
a probability distribution. The zero marker must not make that case look like an initially
uniform population. Keep the existing failure behaviour during this cleanup unless a
separate failure-policy change is agreed.

### Scope recommendation

Resolve the public combination operation and weight-history precision before implementation.
Proceed with explicit internal markers and supported Float32/Float64/BigFloat plus AD tests.
Treat initialisation and constant resolution as local typed operations, without forcing
users to specify a weight type. Keep resampling-gradient estimators and general workspace
widening outside this change, and state those limits when describing AD support.

## Implementation notes

The internal marker names remain `TypelessZero` and `TypelessBaseline`, but neither is a
`Number`. Numeric conversion and promotion overloads have been removed. `add_logweight`
is exported for custom particle updates. Private helpers resolve normalisers only at the
weight-bookkeeping boundary. Ordinary particle steps, APF steps and the CSMC ancestor
sampling path check completed weight types after the first step. History insertion checks
its established state and weight types before mutation.

The primal Float16 count logarithm is explicitly evaluated in Float64 and rounded back,
to avoid overflowing the integer-to-Float16 conversion. This is not a likelihood precision
floor. AD tests target Float32/Float64 and count operations also cover BigFloat. Dual-valued
Float16 with large particle counts is not established as a supported combination.

No resampling derivatives, automatic marker conversions, silent late-step weight casts,
or new numerical-failure rejection policies are introduced by this patch.

### Independent validation

Independent review checked the APF formula against a direct numeric calculation for marked
and numeric initial clouds, Float32/Float64/BigFloat, and nested ForwardDiff. All 24 checks
passed. The completed-step type guard is inferred and allocates zero bytes on a stable path.
The focused package tests cover 143 assertions, including reusable Mooncake preparation,
first-step state/weight promotion, rejection of later changes, and Float32 CSMC across all
three refreshment strategies. Separate APF/resampler regressions passed 171 assertions,
and CUDA 6 runtime tests passed 100 assertions on hardware.

The direct AD reference uses a fixed two-particle cloud with different states, deterministic
parameter-dependent transitions and two observation updates. It compares the filter's
accumulated likelihood and derivatives with the explicit mixture of complete path weights.
This does not rely on differentiating a random ancestor selection.

### Public filter follow-up

Review also reproduced a pre-existing Mooncake failure in the top-level Float32 filter:
its empty-data `0.0` return caused a Float32/Float64 return-type union. The user approved
requiring nonempty observations in `filter`, consistent with the Kalman likelihood.
Empty input is rejected before initialisation. Independent public-filter probes now pass
Float32 and Float64 with dynamic observation vectors, and compare Mooncake with ForwardDiff.
The package regression compares both against the explicit deterministic-path likelihood.
No precision floor or AD-specific rule is introduced.

The full CPU run passed 1,239 checks, with three failures from an in-flight CSMC closure
variable-capture bug. After correcting the local increment name, all 233 APF/CSMC checks
passed in a fresh process. The focused weight suite passed 143 checks, including the 18
Float32 CSMC checks added after the full run began. The subsequent public-filter contract
has separate regression coverage. Keep CI as the final check on the consolidated commit.
