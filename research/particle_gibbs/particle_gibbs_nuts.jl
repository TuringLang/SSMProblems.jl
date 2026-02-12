"""
Manual Particle Gibbs with AdvancedHMC NUTS for a linear-Gaussian SSM with unknown drift.

Same model as particle_gibbs.jl but uses a hand-rolled Gibbs loop instead of Turing:
  - NUTS (via AdvancedHMC) for the drift parameter b
  - Conditional SMC (via GeneralisedFilters) for the latent trajectory x

This gives direct access to HMC diagnostics (acceptance rate, step size, tree depth, etc.).

Model:
    x_t = a * x_{t-1} + b + ε_t,   ε_t ~ N(0, q²)
    y_t = x_t + η_t,               η_t ~ N(0, r²)
    x_0 ~ N(0, σ₀²),   b ~ N(0, σ_b²)
"""

using GeneralisedFilters
using SSMProblems
using AdvancedHMC
using ForwardDiff
using PDMats
using LinearAlgebra
using Random
using StableRNGs
using OffsetArrays
using StatsBase: weights as sb_weights, sample as sb_sample
using Statistics
using Distributions

## ── SSM Construction ─────────────────────────────────────────────────────────

function build_ssm(a, q², r², σ₀², drift)
    return create_homogeneous_linear_gaussian_model(
        [0.0], PDMat([σ₀²;;]), [a;;], [drift], PDMat([q²;;]), [1.0;;], [0.0], PDMat([r²;;])
    )
end

## ── CSMC Step ────────────────────────────────────────────────────────────────

function run_csmc(rng, algo, ssm, ys, ref_traj)
    cb = GeneralisedFilters.DenseAncestorCallback(nothing)
    pf_state, ll = GeneralisedFilters.filter(
        rng, ssm, algo, ys; ref_state=ref_traj, callback=cb
    )
    ws = sb_weights(pf_state)
    idx = sb_sample(rng, 1:(algo.N), ws)
    new_traj = GeneralisedFilters.get_ancestry(cb.container, idx)
    return new_traj, ll
end

## ── Log Density for b ────────────────────────────────────────────────────────
#
# p(b | x, y) ∝ p(b) ∏_t p(x_t | x_{t-1}, b)
# Observations drop out because p(y_t | x_t) does not depend on b.

function make_logdensity(a, q², σ_b², x_traj_ref::Ref, T_len)
    function ℓπ(b_vec)
        b = b_vec[1]
        x = x_traj_ref[]
        ll = logpdf(Normal(0, sqrt(σ_b²)), b)
        for t in 1:T_len
            ll += logpdf(Normal(a * only(x[t - 1]) + b, sqrt(q²)), only(x[t]))
        end
        return ll
    end

    function ∂ℓπ(b_vec)
        v = ℓπ(b_vec)
        g = ForwardDiff.gradient(ℓπ, b_vec)
        return v, g
    end

    return ℓπ, ∂ℓπ
end

## ── Ground Truth via Augmented-State Kalman Filter ───────────────────────────

function augmented_kf_posterior(ys, a, q², r², σ₀², σ_b²)
    A_aug = [a 1.0; 0.0 1.0]
    b_aug = [0.0, 0.0]
    Q_aug = PDMat(Symmetric([q² 0.0; 0.0 1e-12]))
    H_aug = [1.0 0.0]
    c_aug = [0.0]
    R_aug = PDMat([r²;;])
    μ0_aug = [0.0, 0.0]
    Σ0_aug = PDMat(Symmetric([σ₀² 0.0; 0.0 σ_b²]))

    aug_model = create_homogeneous_linear_gaussian_model(
        μ0_aug, Σ0_aug, A_aug, b_aug, Q_aug, H_aug, c_aug, R_aug
    )
    state, ll = GeneralisedFilters.filter(aug_model, KF(), ys)
    return state, ll
end

## ── Run ──────────────────────────────────────────────────────────────────────

rng = StableRNG(42)

# Model parameters
a = 0.8
q² = 0.1
r² = 0.5
σ₀² = 1.0
σ_b² = 4.0
T_len = 10
N_particles = 50
N_iter = 50_000
N_adapts = min(div(N_iter, 10), 1_000)

# Generate data
true_b = 1.5
true_ssm = build_ssm(a, q², r², σ₀², true_b)
_, _, ys = SSMProblems.sample(rng, true_ssm, T_len)

# Ground truth
kf_state, _ = augmented_kf_posterior(ys, a, q², r², σ₀², σ_b²)
println("=== Augmented KF posterior of b ===")
println("  Mean: $(kf_state.μ[2])")
println("  Std:  $(sqrt(kf_state.Σ[2, 2]))")
println("  True: $true_b")

# ── Initialise CSMC ──────────────────────────────────────────────────────────

pf_algo = BF(N_particles)
init_ssm = build_ssm(a, q², r², σ₀², 0.0)
x_traj, _ = run_csmc(rng, pf_algo, init_ssm, ys, nothing)

# ── Initialise NUTS via AdvancedHMC ──────────────────────────────────────────

x_traj_ref = Ref(x_traj)
ℓπ, ∂ℓπ = make_logdensity(a, q², σ_b², x_traj_ref, T_len)

b_vec = [0.0]
metric = DiagEuclideanMetric(1)
hamiltonian = Hamiltonian(metric, ℓπ, ∂ℓπ)

initial_ϵ = find_good_stepsize(hamiltonian, b_vec)
integrator = Leapfrog(initial_ϵ)
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

# ── Gibbs Loop ───────────────────────────────────────────────────────────────

b_samples = Vector{Float64}(undef, N_iter)
acceptance_rates = Vector{Float64}(undef, N_iter)
step_sizes = Vector{Float64}(undef, N_iter)
tree_depths = Vector{Int}(undef, N_iter)

hamiltonian, nuts_trans = AdvancedHMC.sample_init(rng, hamiltonian, b_vec)

for i in 1:N_iter
    # ── NUTS step for b ──────────────────────────────────────────────────
    # Recompute the phase point since the target changed (x_traj updated)
    hamiltonian, nuts_trans = AdvancedHMC.sample_init(rng, hamiltonian, b_vec)
    nuts_trans = AdvancedHMC.transition(rng, hamiltonian, kernel, nuts_trans.z)
    tstat = AdvancedHMC.stat(nuts_trans)

    # Adapt step size and mass matrix during warmup
    hamiltonian, kernel, _ = AdvancedHMC.adapt!(
        hamiltonian, kernel, adaptor, i, N_adapts, nuts_trans.z.θ, tstat.acceptance_rate
    )

    b_vec = nuts_trans.z.θ

    # Record diagnostics
    b_samples[i] = b_vec[1]
    acceptance_rates[i] = tstat.acceptance_rate
    step_sizes[i] = tstat.step_size
    tree_depths[i] = tstat.tree_depth

    # ── CSMC step for x | b ──────────────────────────────────────────────
    current_ssm = build_ssm(a, q², r², σ₀², b_vec[1])
    x_traj, _ = run_csmc(rng, pf_algo, current_ssm, ys, x_traj)
    x_traj_ref[] = x_traj

    if i % 10_000 == 0
        post_warmup = b_samples[max(N_adapts + 1, 1):i]
        println(
            "[$i] b mean=$(round(mean(post_warmup); digits=3)), " *
            "acc=$(round(mean(acceptance_rates[1:i]); digits=3)), " *
            "ϵ=$(round(step_sizes[i]; digits=4))",
        )
    end
end

# ── Results ──────────────────────────────────────────────────────────────────

burn = N_adapts
post_samples = b_samples[(burn + 1):end]
post_acc = acceptance_rates[(burn + 1):end]
post_depths = tree_depths[(burn + 1):end]

println("\n=== Particle Gibbs posterior of b ===")
println("  Mean: $(mean(post_samples))")
println("  Std:  $(std(post_samples))")

println("\n=== Ground truth ===")
println("  KF mean: $(kf_state.μ[2])")
println("  KF std:  $(sqrt(kf_state.Σ[2, 2]))")

println("\n=== NUTS diagnostics (post-warmup) ===")
println("  Acceptance rate: $(round(mean(post_acc); digits=3))")
println("  Final step size: $(round(step_sizes[end]; digits=4))")
println("  Mean tree depth: $(round(mean(post_depths); digits=1))")
println("  Max tree depth:  $(maximum(post_depths))")
