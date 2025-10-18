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

function GeneralisedFilters.will_resample(alt_resampler::AlternatingResampler, state)
    return alt_resampler.resample_next
end

function GeneralisedFilters.resample(
    rng::AbstractRNG,
    alt_resampler::AlternatingResampler,
    state;
    ref_state::Union{Nothing,AbstractVector}=nothing,
)
    alt_resampler.resample_next = !alt_resampler.resample_next
    return GeneralisedFilters.resample(rng, alt_resampler.resampler, state; ref_state)
end
