import LinearAlgebra: I
import Distributions: logpdf
import LogExpFunctions: softmax
import StatsBase: Weights

export RBPF

struct RBPF <: SSMProblems.AbstractFilter
    n_particles::Int
end

struct RaoBlackwellisedParticle
    x::Vector{Float64}
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    log_w::Float64
    parent_idx::Int
end

function filter(
    rng::AbstractRNG,
    model::HierarchicalSSM,
    algo::RBPF,
    observations::AbstractVector,
    extras::AbstractVector,
)
    T = length(observations)
    N = algo.n_particles

    particle_container = Matrix{RaoBlackwellisedParticle}(undef, T, algo.n_particles)

    outer_dyn = model.outer_dyn
    inner_dyn = model.inner_dyn
    obs = model.obs

    for t in 1:T
        y = observations[t]
        u = extras[t]

        if t == 1
            for i in 1:N
                log_w = -log(N)
                parent_idx = 0

                # Initialise outer state from prior
                x = simulate(rng, outer_dyn, u)

                # The control vector for the inner dynamics is the concatenation of the
                # the outer state of the control vector
                inner_u = if isnothing(u)
                    (; new_outer=x)
                else
                    (; u..., new_outer=x)
                end

                # Compute prior mean and covariance for inner state, conditioned on outer
                μ0, Σ0 = calc_initial(inner_dyn, inner_u)

                # Extract model matrices and vectors
                H, c, R = calc_params(obs, t, inner_u)

                # Filter initial states
                K = Σ0 * H' * inv(H * Σ0 * H' + R)
                μ = μ0 + K * (y - H * μ0 - c)
                Σ = (I - K * H) * Σ0

                # Update weight given observation
                μ_y = H * μ0 + c
                Σ_y = H * Σ0 * H' + R
                log_w += logpdf(MvNormal(μ_y, Σ_y), y)

                particle_container[t, i] = RaoBlackwellisedParticle(
                    x, μ, Σ, log_w, parent_idx
                )
            end
        else
            # Resampling
            weights = softmax(getproperty.(particle_container[t - 1, :], :log_w))
            parent_idxs = sample(rng, 1:N, Weights(weights), N)

            for i in 1:N
                # Resample
                parent = particle_container[t - 1, parent_idxs[i]]
                log_w = -log(N)

                # Transition outer state
                x = simulate(rng, outer_dyn, t, parent.x, u)

                # See t = 1 case
                inner_u = if isnothing(u)
                    (; last_outer=parent.x, new_outer=x)
                else
                    (; u..., last_outer=parent.x, new_outer=x)
                end

                # Extract model matrices and vectors
                A, b, Q = calc_params(inner_dyn, t, inner_u)
                H, c, R = calc_params(obs, t, inner_u)

                # Transition inner state
                μ_pred = A * parent.μ + b
                Σ_pred = A * parent.Σ * A' + Q

                # Filter state
                K = Σ_pred * H' * inv(H * Σ_pred * H' + R)
                μ = μ_pred + K * (y - H * μ_pred - c)
                Σ = (I - K * H) * Σ_pred

                # Update weight given observation
                μ_y = H * μ_pred + c
                Σ_y = H * Σ_pred * H' + R
                Σ_y_sym = (Σ_y + Σ_y') / 2
                if maximum(abs.(Σ_y - Σ_y_sym)) > 1e-6
                    println("Σ_y not symmetric")
                    println(Σ_y)
                    println(Σ_y_sym)
                end
                Σ_y = Σ_y_sym
                log_w += logpdf(MvNormal(μ_y, Σ_y), y)

                particle_container[t, i] = RaoBlackwellisedParticle(
                    x, μ, Σ, log_w, parent_idxs[i]
                )
            end
        end
    end

    return particle_container
end
