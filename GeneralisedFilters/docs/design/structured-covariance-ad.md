# Structured covariance parameters and canonical filtering states

Design review, 2026-09-22. The state-storage and AD correctness fixes below are implemented;
the lazy structured-derivative optimisations remain a proposed implementation contract.
This document covers real-valued covariance-form Kalman AD; SRKF has a separate reverse path.

Implementation follow-up: canonical state storage, first-update history promotion, and
mixed-precision likelihood handling are now implemented. Structured covariance AD keeps
Mooncake's derived rules, with a narrow zero-derivative rule for Julia's non-numerical
matrix-wrapper flag. No new handwritten structured covariance VJP is needed for these
correctness fixes. Lazy contractions and first-step prior VJP optimisations remain proposals.
The likelihood accumulator has a Float64 precision floor (preserving wider/AD types) to
avoid the empty/nonempty Float32/Float64 return-type union rejected by Mooncake 0.5.57.

## Decision and corrections to the earlier proposal

Keep `GaussianState{TM<:AbstractVector,TS<:AbstractMatrix}` and structured model parameters.
Canonicalise *computational state storage*, not the covariance parameters in the model.
Make covariance parameter VJPs consume a sensitivity operator rather than requiring a
materialised covariance gradient. Introduce these optimisations incrementally, retaining
an independently tested dense adjoint as a reference and fallback.

Three qualifications are essential:

1. In the current `_kalman_reverse_core`, the process-covariance sensitivity is already
   the predicted-state covariance sensitivity. `_Q_adjoint` returns that matrix; it is
   not another expensive matrix product. Merely specialising `_Q_adjoint` cannot deliver
   the proposed saving. The reverse core and its consumers must be reorganised together.
2. The ordinary state-to-state AD boundary still returns a dense state cotangent. We can
   avoid a separate dense Q gradient, but cannot promise a subquadratic reverse state or
   a linear-cost Kalman gradient. Dense transition products usually remain cubic.
3. Converting the initial prior into a dense state *outside* the first numerical primitive
   exposes a dense cotangent at that boundary. A later optimisation avoiding the dense
   initial-prior covariance gradient requires an internal prior-to-first-step primitive.
   A conversion pullback alone does not achieve that goal.

The public model/container interfaces can remain. A future dimension-bearing isotropic
covariance representation would be an additive public feature; it does not exist today.

## Parameter representation is part of the contract

Preserve the actual Q/R/prior object in each atom until its VJP is evaluated. Do not replace
it with `Matrix(Q)` in model construction or in the numerical primitive's arguments.
A primal calculation may use a materialisation internally while retaining the original
argument; whether materialising is efficient is a separate primal-kernel concern.

| Parameter representation | Independent coordinates | Required parameter VJP for symmetric sensitivity C |
|:--|:--|:--|
| Plain Matrix/SMatrix covariance | Every stored entry; kernel symmetrises | Materialise C in matching tangent storage |
| `Diagonal(q)` | Entries of q | `diagonal(C)`, computed directly |
| Dimension-bearing isotropic covariance s I | Scalar s | `trace(C)`, computed directly |
| `Symmetric(M, :U/:L)` | Selected triangle of M | Diagonal C and twice the selected off-diagonal entries; zero unused triangle |
| `CovarianceFactor(F)` representing FF' | Entries of F | `2 * apply(C, F)` |

`q` here denotes variances. If users build `Diagonal(exp.(theta))`, ordinary AD through
that construction supplies the exponential chain rule; the Kalman rule must not apply it
again. Likewise a standard-deviation parameter needs its own ordinary square chain rule.

`Diagonal(fill(s, n))` works through a vector VJP followed by the fill pullback, but does
not expose a scalar parameter directly at the Kalman boundary. `s * I` is UniformScaling,
not AbstractMatrix, and is not presently accepted by the atoms. Direct scalar trace
specialisation needs a dimension-bearing representation. `s * SMatrix(I)` has already
lost that representation and must use the full-array path. Do not infer structure from
numerical zeros: an unconstrained matrix which happens to be diagonal is not a diagonal
parameterisation.

The Symmetric case generally still has quadratic parameter storage and work. A dense
parent tangent is legitimate there; avoid confusing its stored-triangle semantics with
the averaging operation used for an unconstrained covariance array.

## Explicit reverse algebra

Let n be state dimension and m observation dimension. At one step:

    mu = A mu0 + b
    P  = A P0 A' + Q
    v  = y - H mu - c
    S  = H P H' + R
    W  = inv(S)                    # notation; solve-based implementations are allowed
    K  = P H' W
    J  = I - K H
    w  = W v

The primal continues to use the Joseph covariance update. Let a be the filtered-mean
cotangent, B the symmetric part of the filtered-covariance cotangent, and lambda the
log-likelihood cotangent. B must be symmetrised even for asymmetric output seeds.
Define:

    k = K' a
    r = a - H' k                  # J' a
    u = H' w
    E = lambda/2 * (w w' - W)

The symmetric predicted-covariance sensitivity C and observation-noise sensitivity D are:

    C = J' B J + sym(r u') + lambda/2 * (u u' - H' W H)
    D = K' B K - sym(k w') + E

These describe the mathematical Joseph-form map at the optimal Kalman gain. Algebraic
cancellations are exact in real arithmetic, not necessarily in floating point. Retain the
Joseph primal, compare against differentiation of that actual primal, and test conditioning
before selecting the new reverse path. Do not silently replace a numerically safer path
with one validated only on well-conditioned examples.

### Products without forming C or D

For a matrix X of the appropriate height:

    JX = X - K * (H * X)
    Z  = B * JX
    apply_C(X) = Z - H' * (K' * Z)
                 + (r * (u' * X) + u * (r' * X))/2
                 + lambda/2 * (u * (u' * X) - H' * (W * (H * X)))

    apply_D(X) = K' * (B * (K * X))
                 - (k * (w' * X) + w * (k' * X))/2
                 + lambda/2 * (w * (w' * X) - W * X)

There is no need to form J either. Products with W may be solves against the cached
innovation factor. The current kernel already stores W; removing it is a separate change.

### Direct diagonal and trace contractions

Set U = B K, Z = K' U, and let `coldot(X,Y)` be columnwise inner products:

    diag_C = diag(B) - 2 * coldot(U', H) + coldot(H, Z*H)
             + r .* u + lambda/2 * (u.^2 - coldot(H, W*H))

    diag_D = coldot(K, U) - k .* w + lambda/2 * (w.^2 - diag(W))

`trace_C` and `trace_D` use the corresponding scalar reductions directly; a vector output
is not required. These formulas never form C or D. Reuse U/Z only when consumers need
them; a diagonal-only Q path need not materialise Z if an alternative contraction is
cheaper for the dimensions in question. The initial implementation can use the explicit
formulas above, then benchmark alternatives.

### The remaining reverse pass

All consumers must be accounted for; optimising Q in isolation is insufficient:

    ybar  = k - lambda*w
    mubar = r + lambda*u
    CA    = apply_C(A)
    P0bar = sym(A' * CA)
    Abar  = mubar * mu0' + 2 * CA * P0
    mu0bar = A' * mubar
    bbar  = mubar
    cbar  = -ybar
    T     = E - sym(k*w')
    Hbar  = -ybar*mu' + w*(P*a)' + 2*T*H*P - 2*K'*B*J*P

Hbar may still use a dense m-by-m T. That does not materialise C or D, but it means a
structured R alone does not eliminate all dense observation-space work. Its products can
subsequently be evaluated from factors. Do not force that additional optimisation into
the first implementation.

If A is inactive, P0bar may still require CA. Similarly inactive Q/R do not make C/D's
underlying state/observation intermediates inactive. Activity flags may skip parameter
VJPs, never an intermediate needed by another path. A, H and observation gradients remain
fully supported; do not assume a likelihood-only output seed or fixed observations.

For an isotropic/diagonal/factored Q or R, dispatch to the matching contraction. For a full
array, materialise the operator. Share materialisations/products within a reverse call
when several consumers need them. Avoid a global dense-gradient cache which would make
all supposedly lazy paths allocate their dense sensitivities anyway.

### Cost and memory claims

U = B K costs O(n^2 m), and Z and its products cost O(n m^2). The direct Q diagonal uses
these plus reductions, without an n-by-n C. R's diagonal needs U but neither Z nor an
m-by-m D. The state propagation still produces an n-by-n P0bar, and apply_C(A) with a
fully dense A generally needs O(n^3) work. A rank-r factor VJP returns n-by-r or m-by-r
storage but still acts on dense state sensitivities. None of these is a claim of O(n)
total gradient evaluation, or an automatic speedup when n=m and n is tiny.

Use a dense path when benchmarks justify it. For tiny StaticArrays, operator bookkeeping
or duplicate products may cost more than materialising a small matrix. Keep strategies
concrete and compile-time-specialised; no abstract-function fields, Dict of intermediates,
or retained sensitivity-operator history per time step.

## State storage and scalar types

Normalisation means storage conversion, not covariance repair. It must neither add noise
nor silently reinterpret a Symmetric parent by averaging its unused triangle.

Introduce one internal canonical-state construction path used by initialisation,
likelihood evaluation and numerical step outputs. For the initial supported cases:

- A static mean supplies the compile-time state dimension; produce SVector/SMatrix state
  storage, including from Diagonal(SVector) and Symmetric(SMatrix) inputs.
- A dynamic mean uses Vector/Matrix storage. Initial structured covariance values are
  materialised into that storage; the original prior retains its structured parameter.
- Define and validate the conversion of a dynamic covariance with a static mean explicitly:
  copy into the mean's statically known shape after checking dimensions. This does not
  convert the model parameter or imply the dynamic-parameter reverse path is specialised.
- Promote scalar types from both mean and covariance when constructing a state. Never
  force covariance entries into the mean's original scalar type: that could discard
  precision or ForwardDiff Dual dependence. Preserve static dimensions independently
  of scalar promotion. Never differentiate shape, triangle selectors or storage policy.

A one-time prior conversion alone cannot guarantee history type stability. Parameters or
observations can promote the scalar type on the first step; mixed dense/static operations
can change the representation. Establish history storage from the first filtered state, promoting the stored first
prediction into that representation too: the first prediction itself may still be Float32
before the observation promotes subsequent predictions to Float64. Never allocate histories
from the unpromoted prior or first prediction. Separate the first likelihood step from the subsequent loop with a function barrier so
Float32-to-Float64 promotion does not force a union-typed state/accumulator into Mooncake.
The T=0 path still returns zero likelihood without evaluating any step.

For the fast typed-history path, require stable dimensions and concrete canonical storage
and promoted scalar type after the first step. Validate later results and report an
informative contract error if they change; do not silently downcast into the allocated
history. Supporting arbitrary later type changes with heterogeneous histories is outside
this optimisation. This restriction and the conversion policy must be documented and
checked against existing model examples before adoption. Resolve a callable only once per
step; do not pre-scan all future components to select a type.

For the ordinary first implementation, the prior conversion's pullback receives a dense
state sensitivity and projects it correctly. To avoid that dense *prior* sensitivity too,
add a private `initial_kalman_step(prior, dyn, obs, y)` primitive later, enclosing conversion
and the first step. Its prior sensitivity is the operator X -> A' * apply_C(A*X). Its
Diagonal/trace/factor VJPs can use this congruence without forming A' C A; computing its
diagonal still costs dense matrix work for a general dense A. Keep the public likelihood
interface unchanged and compare the first-step primitive against the ordinary path.

## AD boundary and aliasing

Refactor engine-independent numerical sensitivities separately from Mooncake tangent
packing. Proposed private operations (names illustrative, not a new public plugin API):

    update_sensitivities(cache, mean_seed, covariance_seed, likelihood_seed)
    sensitivity_apply(sensitivity, X)
    sensitivity_diagonal(sensitivity)
    sensitivity_trace(sensitivity)
    sensitivity_materialise(sensitivity)
    covariance_parameter_vjp(covariance_representation, sensitivity)

The last operation returns contributions in *independent parameter coordinates*, not
necessarily an AbstractMatrix. A second layer packs those into the exact tangent of the
original Julia object. Supported primitive signatures must list supported representations;
do not broaden the rule to every AbstractMatrix and then assume a `.data` field exists.
Unknown representations may use the generic AD path, with no promise of specialised
performance or support until tested. Never return silent zeros as a fallback.

Mooncake's immutable-static rule currently returns only rdata. That is insufficient for
Diagonal(Vector), Symmetric(Matrix), or structures containing both scalar and array fields.
The adapter must independently:

1. Reconstruct the complete output seed from output fdata and supplied rdata.
2. Accumulate mutable input contributions into existing fdata with additive updates.
3. Return the appropriate immutable rdata, with scalar types matching each primal field.

These steps are not an either/or branch. Compute intermediates before mutating tangent
buffers; do not overwrite or use existing input tangents as scratch. If Q and R share a
buffer, both semantic contributions must accumulate into that buffer. Repeated time-step
uses, repeated likelihood calls, and another objective term must accumulate too. Static
immutable values are handled by normal upstream AD accumulation. Do not deduplicate
contributions merely because two arguments alias. Cached primals must not be mutated
between forward and reverse evaluation.

Read the actual Mooncake fdata/rdata types rather than inferring inactivity from
`NoRData`: arrays have NoRData but can have active mutable fdata. Structural flags are
non-differentiable, but real numerical parameters such as jitter have real derivatives.
Calculate with output/cached-intermediate precision, then pack each cotangent into its
required input tangent type. Mixed-precision full loops need tests in addition to one-step
rules. These are precisely the failure classes highlighted by issue #188.

Repair remains outside the fused step and sends its pullback into B. Do not fold clipping
or jitter into the structured covariance algebra. SRKF's QR/factor reverse path must not
be silently routed through these covariance-form formulas; give it a separate design when
optimising it. This plan does not relax SRKF rank-change/zero-pivot limitations or change
CSMC covariance-repair validity policy.

## Implementation order and acceptance gates

1. **Storage correctness:** shared state conversion, first-step history allocation and
   scalar-promotion barrier. Reproduce the Symmetric/Diagonal smoother failures and the
   mixed-precision likelihood failure before fixing them. Verify values against dense
   inputs; add ForwardDiff and reverse tests for normalisation, including ignored triangles.
2. **Representation-correct AD:** split tangent packing from algebra; add explicit tested
   covariance representation support, initially Diagonal with static storage, then mutable
   storage and Symmetric. It is acceptable to use dense reference sensitivities here.
3. **Direct parameter VJPs:** introduce the operator formulas and diagonal/trace/factor
   contractions. Verify no materialisation path is invoked in the structured specialisations;
   retain a selectable dense reference path. Do not remove the current adjoint first.
4. **Prior optimisation and additional structures:** optional first-step primitive,
   dimension-bearing isotropic representation, low-rank factor support in the KF rule.
   Rank-deficient process covariance is compatible with KF if the innovation is PD; factor
   derivatives are polynomial, but do not conflate this with differentiating a Cholesky/QR
   factor at a rank change. Observation covariance must meet the algorithm's own contract.
5. **Integration/performance:** full package/AD/Turing tests and targeted benchmarks before
   changing the default strategy. Plain static models must not lose their existing hot-path
   allocation behaviour. Keep optimisations opt-in/internal until demonstrated useful.

Required tests go beyond positive-definite symmetric output seeds:

- Asymmetric, zero and nonzero covariance seeds; mixed signs of likelihood seed; nonzero
  mean seed; all nine input VJPs; n less than, equal to and greater than m; T=0/1/many.
- Direct diagonal/isotropic/factor VJPs versus dense reference, ForwardDiff, and directional
  finite differences of the Joseph primal. Include log-variance parameter construction.
- Symmetric upper/lower storage with deliberately unequal unused triangle; zero derivative
  for ignored entries. A plain unconstrained matrix needs averaging semantics instead.
- Shared Q/R/prior parent arrays, sharing across time, pre-existing nonzero tangents,
  `f(theta)+sum(theta)`, `f(theta)+f(theta)`, reused gradient caches with changed theta/path.
- Float32, Float64, mixed scalar fields, forward Dual fields, dense/static mixing, and
  constant/active repair parameters. Mooncake debug-mode/type and `TestUtils.test_rule`
  checks for every new primitive signature and mixed fdata/rdata object.
- Non-diagonal A/H with diagonal prior/noise, so the state actually becomes dense; history
  and RB CSMC AS/BS tests. Test documented errors for later storage/type changes.
- Ill-conditioned but valid innovations, with high-precision and generic-AD comparisons;
  no changed target or covariance repair to make a failing comparison pass.
- Benchmarks over n=2,8,32,128, several m/n and factor ranks, short/long T, Q/R-only and
  all-parameter activity. Measure compilation, warm time, allocations and peak retained
  memory separately. Measure the complete conditional trajectory objective, not just the
  diagonal extraction. Many-parameter HMC and changed-path Turing cache refresh must be
  covered. No speedup claim until these measurements exist.

## Evidence and limits of this review

The accompanying `check_structured_covariance.jl` passed **600/600 checks on Julia
1.12.7**. It is a standalone algebra experiment,
not a production implementation or part of the ordinary test suite. It checks operator
products, direct diagonals, factor contractions and directional derivatives for every
input against a Joseph-form primal on 40 random well-conditioned systems with asymmetric
output covariance seeds. It does not establish numerical stability near singularities,
Mooncake integration correctness, alias behaviour or a performance improvement.

Relevant implementation: `src/kernels/kalman_adjoint.jl`, `src/kernels/kalman.jl`,
`src/algorithms/kalman.jl`, `src/gaussian.jl`, `ext/MooncakeExt.jl`, and `src/activity.jl`.
Mooncake tangent contracts were checked against installed version 0.5.57, especially
`docs/src/understanding_mooncake/rule_system.md` and `docs/src/utilities/debug_mode.md`.
See also [issue #188](https://github.com/TuringLang/SSMProblems.jl/issues/188) and
[PR #189](https://github.com/TuringLang/SSMProblems.jl/pull/189).

Correctness follow-up validation (2026-09-22): the complete bounds-checked CPU/Turing suite
passed 1,028 assertions, with the existing single Aqua persistent-task skip for the local,
unreleased SSMProblems dependency. All 100 CUDA assertions and the strict documentation
build passed. The existing static benchmark retained 0-byte particle prediction and
conditional objective evaluations, and 80/160-byte prepared ForwardDiff/Mooncake gradients.
These checks validate the storage/promotion and ordinary structured-AD fixes, not the lazy
sensitivity operators proposed above.
