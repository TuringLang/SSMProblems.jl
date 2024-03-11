"""
A Rao-Blackwellised particle filter for conditionally linear Gaussian state space models.

Comments:
- This specific implementation suffers from duplication of the full model dynamics
  between the conditioning and conditional model. This would not occur for a general
  model so is something that needs fixing urgently.
"""

using Distributions
using DynamicIterators
using GaussianDistributions
using Kalman
using LinearAlgebra
using LogExpFunctions
using Plots
using Random
using StatsBase
using ProgressMeter

using Revise

includet("inference.jl")
includet("model.jl")

# Parameters
SEED = 1235
T = 20
N_PARTICLES = 10^5

# Model definition
x0 = [0.01, -0.01]
P0 = Matrix(0.1I, 2, 2)
Φ = [
    -0.2 0.0
    0.5 -0.3
]
b = zeros(2)
Q = [
    0.3 0.0
    0.0 0.2
]
H = [0.0 1.0]
R = Matrix(0.1I, 1, 1)
D1 = 1
D2 = 1

######################
#### GROUND TRUTH ####
######################

# Model definition using Kalman.jl
E = LinearEvolution(Φ, Gaussian(b, Q))
Obs = LinearObservationModel(H, R)
M = LinearStateSpaceModel(E, Obs)

O = LinearObservation(LinearEvolution(Φ, Gaussian(b, Q)), H, R)

# Simulate from model
Random.seed!(SEED)
G0 = Gaussian(x0, P0)
x = rand(StateObs(G0, M.obs))
X = trace(DynamicIterators.Sampled(M), 1 => x, endtime(T))
Y = collect(t => y for (t, (x, y)) in pairs(X))
ys = stack(collect(y for (t, (x, y)) in pairs(X)))

# Ground truth smoothing
Xf, ll = kalmanfilter(M, 1 => G0, Y)
Xs, ll = rts_smoother(M, 1 => G0, Y)

###########################################
#### RAO-BLACKWELLISED PARTICLE FILTER ####
###########################################

m1 = NonAnalyticLinearGaussianSSM(D1, D2, x0, P0, Φ, b, Q, H, R)
m2 = FullyLinearGaussianSubsetSSM(D1, D2, x0, P0, Φ, b, Q, H, R)
model = RaoBlackwellisedSSM(m1, m2)

rng = MersenneTwister(SEED)
particles = filter(rng, model, ys, N_PARTICLES);

####################
#### VALIDATION ####
####################

# Compare inner means
kalman_filter_means = map(G -> G.μ[2], Xf.x)
particle_filter_means = Vector{Float64}(undef, T)
for t in 1:T
    weights = softmax(map(p -> p.log_w, particles[:, t]))
    particle_filter_means[t] = sum(weights .* map(p -> p.μ[], particles[:, t]))
end
p1 = plot(
    collect(1:T),
    kalman_filter_means;
    label="Kalman filter",
    xlabel="Time",
    ylabel="Inner state",
    legend=:bottomleft,
    lw=3,
)
plot!(p1, collect(1:T), particle_filter_means; label="Particle filter", s=:dash, lw=3)

# Compare filtered outer means
kalman_filter_means = map(G -> first(G.μ), Xf.x)
particle_filter_means = Vector{Float64}(undef, T)
for t in 1:T
    weights = softmax(map(p -> p.log_w, particles[:, t]))
    particle_filter_means[t] = sum(weights .* map(p -> p.x[], particles[:, t]))
end
p2 = plot(
    collect(1:T),
    kalman_filter_means;
    label="Kalman filter",
    xlabel="Time",
    ylabel="Outer state",
    legend=:bottomleft,
    lw=3,
)
p2 = plot!(collect(1:T), particle_filter_means; label="Particle filter", s=:dash, lw=3)

plot(p1, p2; layout=(2, 1), size=(800, 600))
