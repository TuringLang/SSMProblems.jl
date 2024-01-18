using Random
using Distributions
using Plots
using LinearAlgebra

sigma_e = 2
sigma_t = 1e-7
F = [
  1. 0. 1. 0.;
  0. 1. 0. 1.;
  0. 0. 1. 0.;
  0. 0. 0. 1.;
]

Z = sigma_t * 1.0I(2)
mu = zeros(2)

function f(beta::AbstractArray)
  atan(beta[2]/beta[1])
end

seed = 2
N = 50
rng = Random.MersenneTwister(seed)
beta = zeros(4, N)
beta[3:4, 1] = rand(rng, MvNormal(mu, Z))
obs = zeros(N)
obs[1] = f(beta[:, 1]) + rand(rng, Normal(0., sigma_e))
for t in 2:N
  beta[:, t] = F * beta[:, t-1] 
  beta[3:4, t] += rand(rng, MvNormal(mu, Z))
  obs[t] = f(beta[:, t-1]) + rand(rng, Normal(0., sigma_e)) 
end
