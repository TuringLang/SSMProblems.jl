"""
    AlternatingResampler

A resampler wrapper that alternates between resampling and not resampling on each step.
This is useful for testing the validity of filters in both cases.

The resampler maintains internal state to track whether it should resample on the next call.
By default, it resamples on the first call, then skips the second, then resamples on the
third, and so on.
"""
mutable struct AlternatingResampler <: GeneralisedFilters.AbstractConditionalResampler
    resampler::GeneralisedFilters.AbstractResampler
    resample_next::Bool
    function AlternatingResampler(
        resampler::GeneralisedFilters.AbstractResampler=Systematic()
    )
        return new(resampler, true)
    end
end

function GeneralisedFilters.resample(
    rng::AbstractRNG,
    alt_resampler::AlternatingResampler,
    state;
    ref_state::Union{Nothing,AbstractVector}=nothing,
)
    n = length(state.particles)

    if alt_resampler.resample_next
        # Resample using wrapped resampler
        alt_resampler.resample_next = false
        return GeneralisedFilters.resample(
            rng, alt_resampler.resampler, state; ref_state
        )
    else
        # Skip resampling - keep particles with their current weights and set ancestors to
        # themselves
        alt_resampler.resample_next = true
        new_particles = similar(state.particles)
        for i in 1:n
            new_particles[i] = GeneralisedFilters.set_ancestor(state.particles[i], i)
        end
        return GeneralisedFilters.ParticleDistribution(new_particles, state.prev_logsumexp)
    end
end
