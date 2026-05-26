## Validation of the new `_step_forward` / `_step_pullback` interface for KalmanFilter.
##
## - Single Kalman step with random matrices.
## - For each scalar output `ll`, vector output `μ_filt`, matrix output `Σ_filt`, set the
##   cotangent of that output to 1 (everything else 0) and run `_step_pullback`.
## - Compare the returned input gradients against central-difference perturbations of
##   the same input on a closure that reads the chosen output.
## - Also verify that wrapping a parameter in `Fixed` causes its gradient to come back
##   as `NoTangent()` instead of a real matrix.

using GeneralisedFilters
using GeneralisedFilters:
    _step_forward, _step_pullback,
    Fixed, FixedParametric, TimeVarying, TimeVaryingParametric,
    LinearGaussianLatentDynamics, LinearGaussianObservationProcess
using ChainRulesCore: NoTangent
using FiniteDifferences
using Distributions: MvNormal, params
using PDMats: PDMat
using LinearAlgebra: I, Symmetric
using Random: MersenneTwister

const Dx = 3
const Dy = 2

rng = MersenneTwister(20260526)

# Set up plausible matrices
μ_prev = randn(rng, Dx)
Σ_prev = let M = randn(rng, Dx, Dx); PDMat(Symmetric(M*M' + 0.1I)); end
A = randn(rng, Dx, Dx); b = randn(rng, Dx)
Q = let M = randn(rng, Dx, Dx); PDMat(Symmetric(M*M' + 0.1I)); end
H = randn(rng, Dy, Dx); c = randn(rng, Dy)
R = let M = randn(rng, Dy, Dy); PDMat(Symmetric(M*M' + 0.1I)); end
y = randn(rng, Dy)

state_prev = MvNormal(μ_prev, Σ_prev)
filter = KalmanFilter()

## A model where everything is parametric so _step_pullback computes real gradients
dyn_active = LinearGaussianLatentDynamics(
    FixedParametric(_ -> A), FixedParametric(_ -> b), FixedParametric(_ -> Q),
)
obs_active = LinearGaussianObservationProcess(
    FixedParametric(_ -> H), FixedParametric(_ -> c), FixedParametric(_ -> R),
)

dyn_params = (; A, b, Q)
obs_params = (; H, c, R)

state_filt, ll, cache = _step_forward(filter, state_prev, dyn_params, obs_params, y)
μ_filt, Σ_filt_pd = params(state_filt)
Σ_filt = Matrix(Σ_filt_pd)

## Helper: run _step_forward and return one of (ll, μ_filt[i], Σ_filt[i,j]) for FD.
function probe(probe_kind, idx; over_A=A, over_b=b, over_Q=Q,
                                 over_H=H, over_c=c, over_R=R,
                                 over_μ_prev=μ_prev, over_Σ_prev=Σ_prev)
    Σ_prev_pd = over_Σ_prev isa PDMat ? over_Σ_prev :
                PDMat(Symmetric(over_Σ_prev))
    Q_pd = over_Q isa PDMat ? over_Q : PDMat(Symmetric(over_Q))
    R_pd = over_R isa PDMat ? over_R : PDMat(Symmetric(over_R))
    state_in = MvNormal(over_μ_prev, Σ_prev_pd)
    dp = (A=over_A, b=over_b, Q=Q_pd)
    op = (H=over_H, c=over_c, R=R_pd)
    sf, l, _ = _step_forward(filter, state_in, dp, op, y)
    if probe_kind === :ll
        return l
    elseif probe_kind === :μ
        return params(sf)[1][idx]
    else  # :Σ
        return Matrix(params(sf)[2])[idx...]
    end
end

fdm = central_fdm(5, 1)

function fd_grad(pkind, pidx, wrt::Symbol)
    # Wrap probe as a function of the chosen input
    if wrt === :A
        return FiniteDifferences.grad(fdm, m -> probe(pkind, pidx; over_A=m), A)[1]
    elseif wrt === :b
        return FiniteDifferences.grad(fdm, v -> probe(pkind, pidx; over_b=v), b)[1]
    elseif wrt === :Q
        return FiniteDifferences.grad(fdm, m -> probe(pkind, pidx; over_Q=m), Matrix(Q))[1]
    elseif wrt === :H
        return FiniteDifferences.grad(fdm, m -> probe(pkind, pidx; over_H=m), H)[1]
    elseif wrt === :c
        return FiniteDifferences.grad(fdm, v -> probe(pkind, pidx; over_c=v), c)[1]
    elseif wrt === :R
        return FiniteDifferences.grad(fdm, m -> probe(pkind, pidx; over_R=m), Matrix(R))[1]
    elseif wrt === :μ_prev
        return FiniteDifferences.grad(fdm, v -> probe(pkind, pidx; over_μ_prev=v), μ_prev)[1]
    elseif wrt === :Σ_prev
        return FiniteDifferences.grad(fdm, m -> probe(pkind, pidx; over_Σ_prev=m), Matrix(Σ_prev))[1]
    end
end

# Project a cotangent onto only the input we're varying. The pullback returns
# gradients for ALL inputs; the test sums the partial w.r.t. the chosen probe
# (and only that probe) by passing a unit cotangent on the chosen output.
function pullback_grad(pkind, pidx)
    ∂μ = zero(μ_filt)
    ∂Σ = zero(Σ_filt)
    Δll = 0.0
    if pkind === :ll
        Δll = 1.0
    elseif pkind === :μ
        ∂μ = collect(zero(μ_filt))
        ∂μ[pidx] = 1.0
    else
        # Σ_filt is symmetric; off-diagonal probe must use a symmetric cotangent
        # (the analytical formulas assume ∂Σ comes from a symmetric matrix variable).
        ∂Σ = zeros(size(Σ_filt))
        i, j = pidx
        if i == j
            ∂Σ[i, j] = 1.0
        else
            ∂Σ[i, j] = 0.5
            ∂Σ[j, i] = 0.5
        end
    end
    (∂μ_prev, ∂Σ_prev), ∂dyn, ∂obs = _step_pullback(
        filter, (∂μ, ∂Σ), Δll, cache, dyn_active, obs_active
    )
    return (
        A = ∂dyn.A, b = ∂dyn.b, Q = ∂dyn.Q,
        H = ∂obs.H, c = ∂obs.c, R = ∂obs.R,
        μ_prev = ∂μ_prev, Σ_prev = ∂Σ_prev,
    )
end

# Symmetrize the FD gradient of a matrix-valued probe w.r.t. a symmetric PDMat input:
# our analytical formulas treat Σ_prev / Q / R as full matrices, so a "lower-triangular"
# FD gradient would underestimate by a factor of 2 on off-diagonals. We symmetrize FD
# perturbation by averaging.
sym_grad(g) = (g + g') / 2

function check_one(pkind, pidx)
    pb = pullback_grad(pkind, pidx)
    label = "$(pkind)$(pidx === nothing ? "" : pidx)"
    for wrt in (:A, :b, :Q, :H, :c, :R, :μ_prev, :Σ_prev)
        fd = fd_grad(pkind, pidx, wrt)
        ana = getproperty(pb, wrt)
        # Σ_prev / Q / R are PDMat-symmetric inputs; FD as a full-matrix probe doesn't
        # know about that symmetry, so we project both onto the symmetric subspace.
        if wrt in (:Σ_prev, :Q, :R)
            fd = sym_grad(fd); ana = sym_grad(ana)
        end
        ok = isapprox(ana, fd; rtol=1e-5, atol=1e-7)
        println("  $label ← $wrt: $(ok ? "OK" : "FAIL  ana=$(ana)  fd=$(fd)")")
    end
end

println("=== Gradient checks against finite differences ===")
println("-- probe = ll --")
check_one(:ll, nothing)
println("-- probe = μ_filt[1] --")
check_one(:μ, 1)
println("-- probe = Σ_filt[1,2] --")
check_one(:Σ, (1, 2))

## Trait dispatch check: mark A as Fixed, verify NoTangent
println("\n=== Trait dispatch ===")
dyn_mixed = LinearGaussianLatentDynamics(
    Fixed(A),                               # A inactive → expect NoTangent
    FixedParametric(_ -> b),                # b active
    TimeVarying((_, _) -> Q),               # Q inactive (TimeVarying)
)
obs_mixed = LinearGaussianObservationProcess(
    FixedParametric(_ -> H),                # H active
    Fixed(c),                               # c inactive
    FixedParametric(_ -> R),                # R active
)
∂μ = zero(μ_filt); ∂Σ = zero(Σ_filt); Δll = 1.0
_, ∂dyn_m, ∂obs_m = _step_pullback(filter, (∂μ, ∂Σ), Δll, cache, dyn_mixed, obs_mixed)
println("  ∂A (Fixed)            isa NoTangent? ", ∂dyn_m.A isa NoTangent)
println("  ∂b (FixedParametric)  is concrete?  ", !(∂dyn_m.b isa NoTangent))
println("  ∂Q (TimeVarying)      isa NoTangent? ", ∂dyn_m.Q isa NoTangent)
println("  ∂H (FixedParametric)  is concrete?  ", !(∂obs_m.H isa NoTangent))
println("  ∂c (Fixed)            isa NoTangent? ", ∂obs_m.c isa NoTangent)
println("  ∂R (FixedParametric)  is concrete?  ", !(∂obs_m.R isa NoTangent))
