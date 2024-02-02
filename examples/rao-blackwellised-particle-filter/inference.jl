include("model.jl")

# TODO: rewrite using Gaussian
struct RaoBlackwellisedParticle
    x::Vector{Float64}
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    log_w::Float64
    parent_idx::Int64
end

function filter(
    rng::AbstractRNG, model::RaoBlackwellisedSSM, ys::Matrix{Float64}, n_particles::Int
)
    T = size(ys, 2)
    m1, m2 = model.conditioning_model, model.conditional_model

    # Create container for particle histories
    particles = Matrix{RaoBlackwellisedParticle}(undef, n_particles, T)

    # Filter and create initial state
    z_sub = m2.z[(m2.D1 + 1):end]
    P_sub = m2.P[(m2.D1 + 1):end, (m2.D1 + 1):end]
    H_sub = m2.H[:, (m2.D1 + 1):end]

    # TODO: replace with solve
    K = P_sub * H_sub' * inv(H_sub * P_sub * H_sub' + m2.R)
    μ = z_sub + K * (ys[:, 1] - H_sub * z_sub)
    Σ = (I - K * H_sub) * P_sub

    # Reweight
    μ_y = H_sub * z_sub
    Σ_y = H_sub * P_sub * H_sub' + m2.R
    log_w = logpdf(MvNormal(μ_y, Σ_y), ys[:, 1])

    for i in 1:n_particles
        x = transition!!(rng, m1)
        particles[i, 1] = RaoBlackwellisedParticle(x, μ, Σ, -log(n_particles) + log_w, -1)
    end

    # Forward pass
    ProgressMeter.@showprogress for t in 2:T

        # Resample particles
        weights = softmax(map(p -> p.log_w, particles[:, t - 1]))
        parent_idxs = sample(1:n_particles, Weights(weights), n_particles)

        # Step filter
        for j in 1:n_particles
            i = parent_idxs[j]
            # Transition outer state
            x = transition!!(rng, m1, particles[i, t - 1].x)

            # Transition inner state
            Q_sub = m2.Q[(m2.D1 + 1):end, (m2.D1 + 1):end]
            A = m2.Φ[(m2.D1 + 1):end, (m2.D1 + 1):end]
            b = m2.Φ[(D1 + 1):end, 1:D1] * particles[i, t - 1].x + m2.b[(m2.D1 + 1):end]
            μ_pred = A * particles[i, t - 1].μ + b
            Σ_pred = A * particles[i, t - 1].Σ * A' + Q_sub

            # Correct state – assuming observing noisy outer state
            K = Σ_pred * H_sub' * inv(H_sub * Σ_pred * H_sub' + m2.R)
            μ = μ_pred + K * (ys[:, t] - H_sub * μ_pred)
            Σ = (I - K * H_sub) * Σ_pred

            # Update particle weight
            μ_y = H_sub * μ_pred
            Σ_y = H_sub * Σ_pred * H_sub' + m2.R
            log_w = -log(n_particles) + logpdf(MvNormal(μ_y, Σ_y), ys[:, t])

            # Create new particle
            particles[j, t] = RaoBlackwellisedParticle(x, μ, Σ, log_w, i)
        end
    end

    return particles
end
