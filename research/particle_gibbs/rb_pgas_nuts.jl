"""
NUTS within Rao-Blackwellised Particle Gibbs with Ancestor Sampling.

Combines:
  - NUTS (AdvancedHMC) for unknown drift/offset parameters θ = (b_x, b_z, c)
  - Rao-Blackwellised conditional SMC with ancestor sampling for the outer trajectory x_{0:T}
  - Analytical Kalman filter gradients for the NUTS step (no ForwardDiff needed)

Model (pretend-nonlinear outer dynamics, linear-Gaussian inner dynamics):
    x_t = A_x x_{t-1} + b_x + ε_x,    ε_x ~ N(0, Q_x)     [outer, treated as general]
    z_t = A_z z_{t-1} + C x_{t-1} + b_z + ε_z,  ε_z ~ N(0, Q_z)  [inner, Kalman-filtered]
    y_t = H z_t + c + η,                η ~ N(0, R)          [observation]

    x_0 ~ N(μ0_x, Σ0_x),  z_0 ~ N(μ0_z, Σ0_z)
    Priors: b_x ~ N(0, σ²_b I),  b_z ~ N(0, σ²_b I),  c ~ N(0, σ²_c I)

Ground truth via augmented-state Kalman filter with state [x; z; b_x; b_z; c].
"""

using GeneralisedFilters
using SSMProblems
using AdvancedHMC
using PDMats
using StaticArrays
using LinearAlgebra
using Random
using StableRNGs
using OffsetArrays
using Distributions
using StatsBase: weights as sb_weights, sample as sb_sample
using Statistics
using LogExpFunctions
using Plots
using ProgressMeter
using LaTeXStrings
using JLD2
using MCMCDiagnosticTools: ess as mcmc_ess

import SSMProblems: prior, dyn, obs
import GeneralisedFilters: resampler, resample, move
using GeneralisedFilters: RBState, InformationLikelihood

## ── Model Types ─────────────────────────────────────────────────────────────

# Outer prior (generic, not GaussianPrior — so RBPF treats it as non-analytical)
struct OuterPrior{μT<:AbstractVector,ΣT<:AbstractPDMat} <: StatePrior
    μ0::μT
    Σ0::ΣT
end

function SSMProblems.distribution(p::OuterPrior; kwargs...)
    return MvNormal(p.μ0, p.Σ0)
end

# Outer dynamics (generic LatentDynamics, NOT LinearGaussianLatentDynamics)
struct OuterDynamics{AT<:AbstractMatrix,bT<:AbstractVector,QT<:AbstractPDMat} <:
       LatentDynamics
    A::AT
    b::bT
    Q::QT
end

function SSMProblems.distribution(d::OuterDynamics, step::Integer, state; kwargs...)
    return MvNormal(d.A * state + d.b, d.Q)
end

## ── Model Construction ──────────────────────────────────────────────────────

function build_model(
    b_x::AbstractVector, b_z::AbstractVector, c_obs::AbstractVector, fixed::NamedTuple
)
    outer_prior = OuterPrior(fixed.μ0_x, fixed.Σ0_x)
    outer_dyn = OuterDynamics(fixed.A_x, b_x, fixed.Q_x)

    inner_prior = GeneralisedFilters.GFTest.InnerPrior(fixed.μ0_z, fixed.Σ0_z)
    inner_dyn = GeneralisedFilters.GFTest.InnerDynamics(fixed.A_z, b_z, fixed.C, fixed.Q_z)
    inner_obs = HomogeneousLinearGaussianObservationProcess(fixed.H, c_obs, fixed.R)

    return HierarchicalSSM(outer_prior, outer_dyn, inner_prior, inner_dyn, inner_obs)
end

## ── Augmented Kalman Filter Ground Truth ────────────────────────────────────

function augmented_kf_posterior(ys, fixed)
    aug_model = build_augmented_model(fixed)
    state, ll = GeneralisedFilters.filter(aug_model, KF(), ys)
    return state, ll
end

function build_augmented_model(fixed)
    Dx, Dz, Dy = fixed.Dx, fixed.Dz, fixed.Dy
    D_aug = 2 * Dx + 2 * Dz + Dy

    A_aug = zeros(D_aug, D_aug)
    A_aug[1:Dx, 1:Dx] = fixed.A_x
    A_aug[1:Dx, (Dx + Dz + 1):(2Dx + Dz)] = I(Dx)
    A_aug[(Dx + 1):(Dx + Dz), 1:Dx] = fixed.C
    A_aug[(Dx + 1):(Dx + Dz), (Dx + 1):(Dx + Dz)] = fixed.A_z
    A_aug[(Dx + 1):(Dx + Dz), (2Dx + Dz + 1):(2Dx + 2Dz)] = I(Dz)
    A_aug[(Dx + Dz + 1):(2Dx + Dz), (Dx + Dz + 1):(2Dx + Dz)] = I(Dx)
    A_aug[(2Dx + Dz + 1):(2Dx + 2Dz), (2Dx + Dz + 1):(2Dx + 2Dz)] = I(Dz)
    A_aug[(2Dx + 2Dz + 1):end, (2Dx + 2Dz + 1):end] = I(Dy)

    b_aug = zeros(D_aug)

    ε = 1e-12
    Q_aug = zeros(D_aug, D_aug)
    Q_aug[1:Dx, 1:Dx] = Matrix(fixed.Q_x)
    Q_aug[(Dx + 1):(Dx + Dz), (Dx + 1):(Dx + Dz)] = Matrix(fixed.Q_z)
    Q_aug[(Dx + Dz + 1):end, (Dx + Dz + 1):end] = ε * I(Dx + Dz + Dy)
    Q_aug = PDMat(Symmetric(Q_aug))

    H_aug = zeros(Dy, D_aug)
    H_aug[:, (Dx + 1):(Dx + Dz)] = fixed.H
    H_aug[:, (2Dx + 2Dz + 1):end] = I(Dy)
    c_aug = zeros(Dy)

    μ0_aug = zeros(D_aug)
    μ0_aug[1:Dx] = fixed.μ0_x
    μ0_aug[(Dx + 1):(Dx + Dz)] = fixed.μ0_z

    Σ0_aug = zeros(D_aug, D_aug)
    Σ0_aug[1:Dx, 1:Dx] = Matrix(fixed.Σ0_x)
    Σ0_aug[(Dx + 1):(Dx + Dz), (Dx + 1):(Dx + Dz)] = Matrix(fixed.Σ0_z)
    Σ0_aug[(Dx + Dz + 1):(2Dx + Dz), (Dx + Dz + 1):(2Dx + Dz)] = fixed.σ²_b * I(Dx)
    Σ0_aug[(2Dx + Dz + 1):(2Dx + 2Dz), (2Dx + Dz + 1):(2Dx + 2Dz)] = fixed.σ²_b * I(Dz)
    Σ0_aug[(2Dx + 2Dz + 1):end, (2Dx + 2Dz + 1):end] = fixed.σ²_c * I(Dy)
    Σ0_aug = PDMat(Symmetric(Σ0_aug))

    return create_homogeneous_linear_gaussian_model(
        μ0_aug, Σ0_aug, A_aug, b_aug, Q_aug, H_aug, c_aug, PDMat(Matrix(fixed.R))
    )
end

function augmented_rts_smooth_all(rng, ys, fixed)
    aug_model = build_augmented_model(fixed)
    T_len = length(ys)

    cache = GeneralisedFilters.StateCallback(nothing, nothing)
    filtered, ll = GeneralisedFilters.filter(rng, aug_model, KF(), ys; callback=cache)

    smoothed_states = Vector{MvNormal}(undef, T_len)
    smoothed_states[T_len] = filtered
    for t in (T_len - 1):-1:1
        smoothed_states[t] = GeneralisedFilters.backward_smooth(
            dyn(aug_model),
            KF(),
            t,
            cache.filtered_states[t],
            smoothed_states[t + 1];
            predicted=cache.proposed_states[t + 1],
        )
    end

    return smoothed_states, filtered, ll
end

## ── NUTS Log-Density and Analytical Gradient ────────────────────────────────

function make_logdensity_and_gradient(fixed, ys, x_traj_ref::Ref)
    Dx, Dz, Dy = fixed.Dx, fixed.Dz, fixed.Dy
    T_len = length(ys)
    Q_x_inv = inv(fixed.Q_x)

    function ℓπ_and_∂ℓπ(θ_vec)
        b_x = SVector{Dx}(θ_vec[1:Dx])
        b_z = SVector{Dz}(θ_vec[(Dx + 1):(Dx + Dz)])
        c_obs = SVector{Dy}(θ_vec[(Dx + Dz + 1):end])
        x_traj = x_traj_ref[]

        # ── Prior ──
        ll = -0.5 * dot(b_x, b_x) / fixed.σ²_b
        ll += -0.5 * dot(b_z, b_z) / fixed.σ²_b
        ll += -0.5 * dot(c_obs, c_obs) / fixed.σ²_c

        ∂b_x = -b_x / fixed.σ²_b
        ∂b_z = -b_z / fixed.σ²_b
        ∂c = -c_obs / fixed.σ²_c

        # ── Outer transitions: Σ_t log N(x_t; A_x x_{t-1} + b_x, Q_x) ──
        for t in 1:T_len
            residual = x_traj[t] - fixed.A_x * x_traj[t - 1] - b_x
            ll += -0.5 * dot(residual, Q_x_inv * residual)
            ll += -0.5 * logdet(fixed.Q_x)
            ll += -0.5 * Dx * log(2π)
            ∂b_x += Q_x_inv * residual
        end

        # ── Inner Kalman filter: forward pass with gradient caches ──
        inner_dyn = GeneralisedFilters.GFTest.InnerDynamics(
            fixed.A_z, b_z, fixed.C, fixed.Q_z
        )
        inner_obs = HomogeneousLinearGaussianObservationProcess(fixed.H, c_obs, fixed.R)

        state = MvNormal(fixed.μ0_z, fixed.Σ0_z)
        caches = Vector{GeneralisedFilters.KalmanGradientCache}(undef, T_len)
        ll_kf = 0.0

        for t in 1:T_len
            # Predict inner state conditioned on outer trajectory
            state = GeneralisedFilters.predict(
                Random.default_rng(),
                inner_dyn,
                KF(),
                t,
                state,
                nothing;
                prev_outer=x_traj[t - 1],
            )
            # Update with cache
            state, ll_inc, caches[t] = GeneralisedFilters.update_with_cache(
                inner_obs, KF(), t, state, ys[t]
            )
            ll_kf += ll_inc
        end
        ll += ll_kf

        # ── Inner Kalman filter: backward gradient pass ──
        # Computes ∂(-ll_kf)/∂params, so we negate to get ∂ll_kf/∂params
        H = fixed.H
        R = fixed.R
        A_z = fixed.A_z

        ∂μ = zero(b_z)
        ∂Σ = zero(fixed.A_z)
        ∂b_z_nll = zero(b_z)
        ∂c_nll = zero(c_obs)

        for t in T_len:-1:1
            ∂c_nll += GeneralisedFilters.gradient_c(∂μ, caches[t])
            ∂μ_pred, ∂Σ_pred = GeneralisedFilters.backward_gradient_update(
                ∂μ, ∂Σ, caches[t], H, R
            )
            ∂b_z_nll += GeneralisedFilters.gradient_b(∂μ_pred)
            ∂μ, ∂Σ = GeneralisedFilters.backward_gradient_predict(∂μ_pred, ∂Σ_pred, A_z)
        end

        # NLL gradients → LL gradients (negate)
        ∂b_z -= ∂b_z_nll
        ∂c -= ∂c_nll

        ∂θ = Vector(vcat(∂b_x, ∂b_z, ∂c))
        return ll, ∂θ
    end

    function ℓπ(θ_vec)
        ll, _ = ℓπ_and_∂ℓπ(θ_vec)
        return ll
    end

    function ∂ℓπ(θ_vec)
        return ℓπ_and_∂ℓπ(θ_vec)
    end

    return ℓπ, ∂ℓπ
end

## ── MH Log-Density (gradient-free) ───────────────────────────────────────────

function make_logdensity(fixed, ys, x_traj_ref::Ref)
    Dx, Dz, Dy = fixed.Dx, fixed.Dz, fixed.Dy
    T_len = length(ys)
    Q_x_inv = inv(fixed.Q_x)

    function ℓπ(θ_vec)
        b_x = SVector{Dx}(θ_vec[1:Dx])
        b_z = SVector{Dz}(θ_vec[(Dx + 1):(Dx + Dz)])
        c_obs = SVector{Dy}(θ_vec[(Dx + Dz + 1):end])
        x_traj = x_traj_ref[]

        # Prior
        ll = -0.5 * dot(b_x, b_x) / fixed.σ²_b
        ll += -0.5 * dot(b_z, b_z) / fixed.σ²_b
        ll += -0.5 * dot(c_obs, c_obs) / fixed.σ²_c

        # Outer transitions
        for t in 1:T_len
            residual = x_traj[t] - fixed.A_x * x_traj[t - 1] - b_x
            ll += -0.5 * dot(residual, Q_x_inv * residual)
        end

        # Inner Kalman filter forward pass
        inner_dyn = GeneralisedFilters.GFTest.InnerDynamics(
            fixed.A_z, b_z, fixed.C, fixed.Q_z
        )
        inner_obs = HomogeneousLinearGaussianObservationProcess(fixed.H, c_obs, fixed.R)

        state = MvNormal(fixed.μ0_z, fixed.Σ0_z)
        for t in 1:T_len
            state = GeneralisedFilters.predict(
                Random.default_rng(),
                inner_dyn,
                KF(),
                t,
                state,
                nothing;
                prev_outer=x_traj[t - 1],
            )
            state, ll_inc = GeneralisedFilters.update(inner_obs, KF(), t, state, ys[t])
            ll += ll_inc
        end

        return ll
    end

    return ℓπ
end

## ── CSMC-AS with RBPF ──────────────────────────────────────────────────────

function run_rbcsmc_as(rng, model, rbpf, ys, ref_traj, predictive_likelihoods)
    N = rbpf.pf.N
    K = length(ys)

    bf_state = initialise(rng, prior(model), rbpf; ref_state=ref_traj)
    particles = bf_state.particles
    container = GeneralisedFilters.DenseParticleContainer(
        OffsetVector([deepcopy(getfield.(particles, :state))], -1),
        Vector{Float64}[],
        Vector{Int}[],
    )

    for t in 1:K
        bf_state = resample(rng, resampler(rbpf), bf_state; ref_state=ref_traj)

        # Ancestor sampling for reference particle
        if !isnothing(ref_traj)
            ref_rb_state = RBState(ref_traj[t], predictive_likelihoods[t])
            ancestor_weights = map(bf_state.particles) do particle
                ancestor_weight(particle, dyn(model), rbpf, t, ref_rb_state)
            end
            ancestor_idx = sb_sample(rng, 1:N, sb_weights(softmax(ancestor_weights)))
        end

        bf_state, _ = move(rng, model, rbpf, t, bf_state, ys[t]; ref_state=ref_traj)

        if !isnothing(ref_traj)
            bf_state.particles[end] = GeneralisedFilters.Particle(
                bf_state.particles[end].state, bf_state.particles[end].log_w, ancestor_idx
            )
        end

        # push!(container.particles, deepcopy(getfield.(bf_state.particles, :state)))
        # push!(container.weights, deepcopy(getfield.(bf_state.particles, :log_w)))
        # push!(container.ancestors, deepcopy(getfield.(bf_state.particles, :ancestor)))
        push!(container.particles, deepcopy(getfield.(bf_state.particles, :state)))
        push!(container.weights, deepcopy(getfield.(bf_state.particles, :log_w)))
        push!(container.ancestors, deepcopy(getfield.(bf_state.particles, :ancestor)))
    end

    # Sample trajectory
    ws = sb_weights(bf_state)
    sampled_idx = sb_sample(rng, 1:N, ws)
    full_traj = GeneralisedFilters.get_ancestry(container, sampled_idx)

    # Extract outer trajectory
    outer_traj = getproperty.(full_traj, :x)

    # Compute backward predictive likelihoods for next iteration
    bip = BackwardInformationPredictor(; initial_jitter=1e-8)
    pred_lik = backward_initialise(rng, model.inner_model.obs, bip, K, ys[K])
    new_pred_liks = Vector{typeof(pred_lik)}(undef, K)
    new_pred_liks[K] = deepcopy(pred_lik)

    for t in (K - 1):-1:1
        pred_lik = backward_predict(
            rng,
            model.inner_model.dyn,
            bip,
            t,
            pred_lik;
            prev_outer=outer_traj[t],
            new_outer=outer_traj[t + 1],
        )
        pred_lik = backward_update(model.inner_model.obs, bip, t, pred_lik, ys[t])
        new_pred_liks[t] = deepcopy(pred_lik)
    end

    return outer_traj, new_pred_liks
end

## ── Generate Data and Ground Truth ──────────────────────────────────────────

rng = StableRNG(42)

# Dimensions
Dx = 2
Dz = 2
Dy = 2

# Fixed model parameters
A_x = @SMatrix [0.8 0.1; -0.1 0.7]
Q_x = PDMat(Symmetric(@SMatrix [0.3 0.05; 0.05 0.2]))
A_z = @SMatrix [0.6 0.15; -0.05 0.5]
C = @SMatrix [0.3 -0.1; 0.1 0.2]
Q_z = PDMat(Symmetric(@SMatrix [0.2 0.03; 0.03 0.15]))
H = @SMatrix [1.0 0.0; 0.5 1.0]
R = PDMat(Symmetric(@SMatrix [0.4 0.05; 0.05 0.3]))
μ0_x = @SVector zeros(2)
Σ0_x = PDMat(Symmetric(@SMatrix [1.0 0.0; 0.0 1.0]))
μ0_z = @SVector zeros(2)
Σ0_z = PDMat(Symmetric(@SMatrix [1.0 0.0; 0.0 1.0]))

# Prior hyperparameters
σ²_b = 4.0
σ²_c = 4.0

fixed = (; Dx, Dz, Dy, A_x, Q_x, A_z, C, Q_z, H, R, μ0_x, Σ0_x, μ0_z, Σ0_z, σ²_b, σ²_c)

# True parameter values
true_b_x = @SVector [1.0, -0.5]
true_b_z = @SVector [0.5, 0.3]
true_c = @SVector [-0.2, 0.4]

T_len = 20
N_particles = 200
N_iter = 10000
N_adapts = min(div(N_iter, 10), 2_000)
# N_adapts = 5

# Generate data from the true model
true_model = build_model(true_b_x, true_b_z, true_c, fixed)
_, _, _, _, ys = sample(rng, true_model, T_len)

# Ground truth
kf_state, kf_ll = augmented_kf_posterior(ys, fixed)
println("=== Augmented KF Posterior ===")
println("  b_x mean: $(kf_state.μ[(Dx + Dz + 1):(2Dx + Dz)])")
println("  b_x std:  $(sqrt.(diag(kf_state.Σ)[(Dx + Dz + 1):(2Dx + Dz)]))")
println("  b_z mean: $(kf_state.μ[(2Dx + Dz + 1):(2Dx + 2Dz)])")
println("  b_z std:  $(sqrt.(diag(kf_state.Σ)[(2Dx + Dz + 1):(2Dx + 2Dz)]))")
println("  c   mean: $(kf_state.μ[(2Dx + 2Dz + 1):end])")
println("  c   std:  $(sqrt.(diag(kf_state.Σ)[(2Dx + 2Dz + 1):end]))")
println("  True b_x: $true_b_x")
println("  True b_z: $true_b_z")
println("  True c:   $true_c")

function run_gibbs(rng, fixed, ys, N_particles, N_iter, N_adapts, traj_thin=50)
    (; Dx, Dz, Dy) = fixed
    θ_dim = Dx + Dz + Dy

    # ── Initialise CSMC-AS ──
    θ_vec = zeros(θ_dim)
    b_x, b_z, c_obs = zero(fixed.μ0_x), zero(fixed.μ0_z), zero(ys[1])

    init_model = build_model(b_x, b_z, c_obs, fixed)
    rbpf = RBPF(BF(N_particles; threshold=0.8), KalmanFilter())
    x_traj, pred_liks = run_rbcsmc_as(rng, init_model, rbpf, ys, nothing, nothing)

    # ── Initialise NUTS ──
    x_traj_ref = Ref(x_traj)
    ℓπ, ∂ℓπ = make_logdensity_and_gradient(fixed, ys, x_traj_ref)

    metric = DiagEuclideanMetric(θ_dim)
    hamiltonian = Hamiltonian(metric, ℓπ, ∂ℓπ)

    initial_ϵ = find_good_stepsize(hamiltonian, θ_vec)
    integrator = Leapfrog(initial_ϵ)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    # ── Storage ──
    b_x_samples = Matrix{Float64}(undef, Dx, N_iter)
    b_z_samples = Matrix{Float64}(undef, Dz, N_iter)
    c_samples = Matrix{Float64}(undef, Dy, N_iter)
    acceptance_rates = Vector{Float64}(undef, N_iter)
    step_sizes = Vector{Float64}(undef, N_iter)
    tree_depths = Vector{Int}(undef, N_iter)
    traj_samples = Vector{typeof(x_traj)}()

    hamiltonian, nuts_trans = AdvancedHMC.sample_init(rng, hamiltonian, θ_vec)

    running_bx = zeros(Dx)
    running_bz = zeros(Dz)
    running_c = zeros(Dy)

    prog = Progress(N_iter; dt=1, desc="RB-PGAS-NUTS")
    for i in 1:N_iter
        # ── NUTS step for θ given x_{0:T} ──
        hamiltonian, nuts_trans = AdvancedHMC.sample_init(rng, hamiltonian, θ_vec)
        nuts_trans = AdvancedHMC.transition(rng, hamiltonian, kernel, nuts_trans.z)
        tstat = AdvancedHMC.stat(nuts_trans)

        hamiltonian, kernel, _ = AdvancedHMC.adapt!(
            hamiltonian, kernel, adaptor, i, N_adapts, nuts_trans.z.θ, tstat.acceptance_rate
        )

        θ_vec = nuts_trans.z.θ
        b_x = SVector{Dx}(θ_vec[1:Dx])
        b_z = SVector{Dz}(θ_vec[(Dx + 1):(Dx + Dz)])
        c_obs = SVector{Dy}(θ_vec[(Dx + Dz + 1):end])

        b_x_samples[:, i] = b_x
        b_z_samples[:, i] = b_z
        c_samples[:, i] = c_obs
        acceptance_rates[i] = tstat.acceptance_rate
        step_sizes[i] = tstat.step_size
        tree_depths[i] = tstat.tree_depth

        running_bx .+= (b_x .- running_bx) ./ i
        running_bz .+= (b_z .- running_bz) ./ i
        running_c .+= (c_obs .- running_c) ./ i

        # ── CSMC-AS step for x_{0:T} given θ ──
        current_model = build_model(b_x, b_z, c_obs, fixed)
        x_traj, pred_liks = run_rbcsmc_as(rng, current_model, rbpf, ys, x_traj, pred_liks)
        x_traj_ref[] = x_traj

        if i > N_adapts && (i - N_adapts) % traj_thin == 0
            push!(traj_samples, deepcopy(x_traj))
        end

        next!(
            prog;
            showvalues=[
                ("mean(b_x)", round.(running_bx; digits=3)),
                ("mean(b_z)", round.(running_bz; digits=3)),
                ("mean(c)", round.(running_c; digits=3)),
                ("accept rate", round(tstat.acceptance_rate; digits=3)),
            ],
        )
    end

    return (;
        b_x_samples,
        b_z_samples,
        c_samples,
        acceptance_rates,
        step_sizes,
        tree_depths,
        traj_samples,
    )
end

function run_gibbs_mh(rng, fixed, ys, N_particles, N_iter, Σ_prop; burn=0, traj_thin=50)
    (; Dx, Dz, Dy) = fixed
    θ_dim = Dx + Dz + Dy

    # Asymptotically optimal RW-MH proposal: (2.38)^2/d * Σ
    prop_cov = (2.38^2 / θ_dim) * Σ_prop
    prop_dist = MvNormal(zeros(θ_dim), Symmetric(prop_cov))

    # ── Initialise CSMC-AS ──
    θ_vec = zeros(θ_dim)
    b_x, b_z, c_obs = zero(fixed.μ0_x), zero(fixed.μ0_z), zero(ys[1])

    init_model = build_model(b_x, b_z, c_obs, fixed)
    rbpf = RBPF(BF(N_particles; threshold=0.8), KalmanFilter())
    x_traj, pred_liks = run_rbcsmc_as(rng, init_model, rbpf, ys, nothing, nothing)

    # ── Initialise MH ──
    x_traj_ref = Ref(x_traj)
    ℓπ = make_logdensity(fixed, ys, x_traj_ref)
    ll_current = ℓπ(θ_vec)

    # ── Storage ──
    b_x_samples = Matrix{Float64}(undef, Dx, N_iter)
    b_z_samples = Matrix{Float64}(undef, Dz, N_iter)
    c_samples = Matrix{Float64}(undef, Dy, N_iter)
    accepted = Vector{Bool}(undef, N_iter)
    traj_samples = Vector{typeof(x_traj)}()

    running_bx = zeros(Dx)
    running_bz = zeros(Dz)
    running_c = zeros(Dy)
    n_accepted = 0

    prog = Progress(N_iter; dt=1, desc="RB-PGAS-MH ")
    for i in 1:N_iter
        # ── MH step for θ given x_{0:T} ──
        θ_proposed = θ_vec + rand(rng, prop_dist)
        ll_proposed = ℓπ(θ_proposed)

        if log(rand(rng)) < ll_proposed - ll_current
            θ_vec = θ_proposed
            ll_current = ll_proposed
            n_accepted += 1
            accepted[i] = true
        else
            accepted[i] = false
        end

        b_x = SVector{Dx}(θ_vec[1:Dx])
        b_z = SVector{Dz}(θ_vec[(Dx + 1):(Dx + Dz)])
        c_obs = SVector{Dy}(θ_vec[(Dx + Dz + 1):end])

        b_x_samples[:, i] = b_x
        b_z_samples[:, i] = b_z
        c_samples[:, i] = c_obs

        running_bx .+= (b_x .- running_bx) ./ i
        running_bz .+= (b_z .- running_bz) ./ i
        running_c .+= (c_obs .- running_c) ./ i

        # ── CSMC-AS step for x_{0:T} given θ ──
        current_model = build_model(b_x, b_z, c_obs, fixed)
        x_traj, pred_liks = run_rbcsmc_as(rng, current_model, rbpf, ys, x_traj, pred_liks)
        x_traj_ref[] = x_traj
        ll_current = ℓπ(θ_vec)

        if i > burn && (i - burn) % traj_thin == 0
            push!(traj_samples, deepcopy(x_traj))
        end

        next!(
            prog;
            showvalues=[
                ("mean(b_x)", round.(running_bx; digits=3)),
                ("mean(b_z)", round.(running_bz; digits=3)),
                ("mean(c)", round.(running_c; digits=3)),
                ("accept rate", round(n_accepted / i; digits=3)),
            ],
        )
    end

    return (; b_x_samples, b_z_samples, c_samples, accepted, traj_samples)
end

nuts_time = @elapsed results_nuts = run_gibbs(rng, fixed, ys, N_particles, N_iter, N_adapts)

## ── NUTS Results ──────────────────────────────────────────────────────────────

nuts_bx = results_nuts.b_x_samples
nuts_bz = results_nuts.b_z_samples
nuts_c = results_nuts.c_samples
nuts_acc = results_nuts.acceptance_rates
nuts_steps = results_nuts.step_sizes
nuts_depths = results_nuts.tree_depths
nuts_trajs = results_nuts.traj_samples

burn = N_adapts
post_nuts_bx = nuts_bx[:, (burn + 1):end]
post_nuts_bz = nuts_bz[:, (burn + 1):end]
post_nuts_c = nuts_c[:, (burn + 1):end]

kf_bx_idx = (Dx + Dz + 1):(2Dx + Dz)
kf_bz_idx = (2Dx + Dz + 1):(2Dx + 2Dz)
kf_c_idx = (2Dx + 2Dz + 1):(2Dx + 2Dz + Dy)

println("\n=== RB-PGAS-NUTS Posterior ===")
println("  b_x mean: $(round.(mean(post_nuts_bx; dims=2)[:]; digits=4))")
println("  b_x std:  $(round.(std(post_nuts_bx; dims=2)[:]; digits=4))")
println("  b_z mean: $(round.(mean(post_nuts_bz; dims=2)[:]; digits=4))")
println("  b_z std:  $(round.(std(post_nuts_bz; dims=2)[:]; digits=4))")
println("  c   mean: $(round.(mean(post_nuts_c; dims=2)[:]; digits=4))")
println("  c   std:  $(round.(std(post_nuts_c; dims=2)[:]; digits=4))")

println("\n=== Augmented KF Ground Truth ===")
println("  b_x mean: $(round.(kf_state.μ[kf_bx_idx]; digits=4))")
println("  b_x std:  $(round.(sqrt.(diag(kf_state.Σ)[kf_bx_idx]); digits=4))")
println("  b_z mean: $(round.(kf_state.μ[kf_bz_idx]; digits=4))")
println("  b_z std:  $(round.(sqrt.(diag(kf_state.Σ)[kf_bz_idx]); digits=4))")
println("  c   mean: $(round.(kf_state.μ[kf_c_idx]; digits=4))")
println("  c   std:  $(round.(sqrt.(diag(kf_state.Σ)[kf_c_idx]); digits=4))")

println("\n=== True Values ===")
println("  b_x: $true_b_x")
println("  b_z: $true_b_z")
println("  c:   $true_c")

println("\n=== NUTS Diagnostics (post-warmup) ===")
println("  Acceptance rate: $(round(mean(nuts_acc[(burn + 1):end]); digits=3))")
println("  Final step size: $(round(nuts_steps[end]; digits=4))")
println("  Mean tree depth: $(round(mean(nuts_depths[(burn + 1):end]); digits=1))")
println("  Max tree depth:  $(maximum(nuts_depths[(burn + 1):end]))")

## ── Run MH-within-PGAS ───────────────────────────────────────────────────────

# Estimate proposal covariance from NUTS posterior
θ_post_nuts = vcat(post_nuts_bx, post_nuts_bz, post_nuts_c)'  # (n_post, θ_dim)
Σ_nuts = cov(θ_post_nuts)

mh_time = @elapsed results_mh = run_gibbs_mh(
    StableRNG(42), fixed, ys, N_particles, N_iter, Σ_nuts; burn=burn
)

mh_bx = results_mh.b_x_samples
mh_bz = results_mh.b_z_samples
mh_c = results_mh.c_samples
mh_accepted = results_mh.accepted
mh_trajs = results_mh.traj_samples

post_mh_bx = mh_bx[:, (burn + 1):end]
post_mh_bz = mh_bz[:, (burn + 1):end]
post_mh_c = mh_c[:, (burn + 1):end]

println("\n=== RB-PGAS-MH Posterior ===")
println("  b_x mean: $(round.(mean(post_mh_bx; dims=2)[:]; digits=4))")
println("  b_x std:  $(round.(std(post_mh_bx; dims=2)[:]; digits=4))")
println("  b_z mean: $(round.(mean(post_mh_bz; dims=2)[:]; digits=4))")
println("  b_z std:  $(round.(std(post_mh_bz; dims=2)[:]; digits=4))")
println("  c   mean: $(round.(mean(post_mh_c; dims=2)[:]; digits=4))")
println("  c   std:  $(round.(std(post_mh_c; dims=2)[:]; digits=4))")
println("  Accept rate: $(round(mean(mh_accepted[(burn + 1):end]); digits=3))")

## ── ESS / Wall-Time Comparison ───────────────────────────────────────────────

param_labels = ["b_x[1]", "b_x[2]", "b_z[1]", "b_z[2]", "c[1]", "c[2]"]

# Stack post-warmup samples: (n_post, n_params) → reshape to (n_post, 1, n_params)
nuts_samples = vcat(post_nuts_bx, post_nuts_bz, post_nuts_c)'
mh_samples = vcat(post_mh_bx, post_mh_bz, post_mh_c)'

nuts_ess = mcmc_ess(reshape(nuts_samples, :, 1, size(nuts_samples, 2)))
mh_ess = mcmc_ess(reshape(mh_samples, :, 1, size(mh_samples, 2)))

println("\n=== ESS / Wall-Time Comparison ===")
println("  NUTS wall time: $(round(nuts_time; digits=1))s")
println("  MH   wall time: $(round(mh_time; digits=1))s")
println()
println("  Parameter   NUTS ESS   MH ESS   NUTS ESS/s   MH ESS/s")
println("  " * "─"^60)
for (i, name) in enumerate(param_labels)
    ne = nuts_ess[i]
    me = mh_ess[i]
    println(
        "  $(rpad(name, 12))" *
        "$(lpad(round(ne; digits=1), 8))   " *
        "$(lpad(round(me; digits=1), 7))   " *
        "$(lpad(round(ne / nuts_time; digits=1), 10))   " *
        "$(lpad(round(me / mh_time; digits=1), 8))",
    )
end
println()
println(
    "  Min ESS/s:  NUTS=$(round(minimum(nuts_ess) / nuts_time; digits=1))  " *
    "MH=$(round(minimum(mh_ess) / mh_time; digits=1))",
)

## ── Comparison Trace Plots ──────────────────────────────────────────────────

param_names = [[L"b_{x,1}", L"b_{x,2}"], [L"b_{z,1}", L"b_{z,2}"], [L"c_1", L"c_2"]]
nuts_all = [post_nuts_bx, post_nuts_bz, post_nuts_c]
mh_all = [post_mh_bx, post_mh_bz, post_mh_c]
kf_indices = [kf_bx_idx, kf_bz_idx, kf_c_idx]

trace_plots = []
for (nuts_s, mh_s, idx, names) in zip(nuts_all, mh_all, kf_indices, param_names)
    for d in 1:2
        kf_mean = kf_state.μ[idx[d]]
        kf_std = sqrt(diag(kf_state.Σ)[idx[d]])
        p = plot(; ylabel=names[d], legend=:topright, size=(800, 200))
        plot!(p, nuts_s[d, :]; label="NUTS", alpha=0.3, linewidth=0.5, color=:steelblue)
        plot!(p, mh_s[d, :]; label="MH", alpha=0.3, linewidth=0.5, color=:darkorange)
        hline!(p, [kf_mean]; color=:red, linewidth=2, label="KF mean")
        hline!(
            p,
            [kf_mean - 1.96 * kf_std, kf_mean + 1.96 * kf_std];
            color=:red,
            linestyle=:dash,
            linewidth=1.5,
            label="95% CI",
        )
        push!(trace_plots, p)
    end
end

p_trace = plot(
    trace_plots...;
    layout=(length(trace_plots), 1),
    size=(900, 200 * length(trace_plots)),
    left_margin=5Plots.mm,
)

## ── ESS/s Bar Chart ─────────────────────────────────────────────────────────

param_names_flat = [L"b_{x,1}", L"b_{x,2}", L"b_{z,1}", L"b_{z,2}", L"c_1", L"c_2"]
nuts_ess_per_s = nuts_ess ./ nuts_time
mh_ess_per_s = mh_ess ./ mh_time

x_pos = collect(1:length(param_names_flat))
bw = 0.35
p_ess = bar(
    x_pos .- bw / 2,
    nuts_ess_per_s;
    bar_width=bw,
    label="NUTS",
    color=:steelblue,
    ylabel="ESS/s",
    title="Effective Samples per Second",
    xticks=(x_pos, param_names_flat),
    size=(700, 400),
    legend=:topright,
)
bar!(p_ess, x_pos .+ bw / 2, mh_ess_per_s; bar_width=bw, label="MH", color=:darkorange)

## ── Smoothed Trajectory Plots ───────────────────────────────────────────────

# Ground truth: RTS smoother on augmented model
smoothed_states, _, _ = augmented_rts_smooth_all(StableRNG(42), ys, fixed)

# Extract smoothed means and 95% CIs for outer state x
ts = 1:T_len
smooth_x_mean = hcat([s.μ[1:Dx] for s in smoothed_states]...)
smooth_x_std = hcat([sqrt.(diag(s.Σ)[1:Dx]) for s in smoothed_states]...)

# Subsample NUTS trajectories for plotting
N_traj_plot = min(200, length(nuts_trajs))
plot_indices = sort(
    sb_sample(StableRNG(123), 1:length(nuts_trajs), N_traj_plot; replace=false)
)

traj_plots = []
for d in 1:Dx
    p = plot(; ylabel=L"x_{%$d}", xlabel="t", legend=:topright, size=(800, 300))

    # Plot sampled trajectories
    for (j, idx) in enumerate(plot_indices)
        traj = nuts_trajs[idx]
        plot!(
            p,
            0:T_len,
            [traj[t][d] for t in 0:T_len];
            color=:steelblue,
            alpha=0.1,
            linewidth=0.5,
            label=(j == 1 ? "PGAS samples" : nothing),
        )
    end

    # RTS smoothed mean and 95% CI
    plot!(p, ts, smooth_x_mean[d, :]; color=:red, linewidth=2.5, label="RTS mean")
    plot!(
        p,
        ts,
        smooth_x_mean[d, :] + 1.96 * smooth_x_std[d, :];
        color=:red,
        linestyle=:dash,
        linewidth=1.5,
        label="95% CI",
    )
    plot!(
        p,
        ts,
        smooth_x_mean[d, :] - 1.96 * smooth_x_std[d, :];
        color=:red,
        linestyle=:dash,
        linewidth=1.5,
        label=nothing,
    )

    push!(traj_plots, p)
end

p_traj = plot(traj_plots...; layout=(Dx, 1), size=(900, 300 * Dx), left_margin=5Plots.mm)
