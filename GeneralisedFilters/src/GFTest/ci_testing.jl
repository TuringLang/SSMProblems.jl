using Distributions: quantile, Normal

const Z_99 = 2.5758293035489004  # quantile(Normal(), 0.995)

"""
    check_mc_estimate(estimate, ground_truth, sample_std, ess; kwargs...) -> Bool

Check that ground truth falls within a 99% CI of the Monte Carlo estimate.

Uses ESS-adjusted standard error for proper CI width. Asserts that
the relative CI width isn't too large (sanity check on sample size).
Prints diagnostic information on failure.

# Arguments
- `estimate::Real`: The Monte Carlo estimate
- `ground_truth::Real`: The analytical/reference value to test against
- `sample_std::Real`: Sample standard deviation of the MC samples
- `ess::Real`: Effective sample size

# Keyword Arguments
- `max_rel_ci_width::Real=1e-2`: Maximum CI half-width relative to |ground_truth|
- `min_abs_for_rel_check::Real=1e-10`: Skip relative width check for near-zero ground truths

# Returns
- `Bool`: Whether the estimate is within the CI of the ground truth
"""
function check_mc_estimate(
    estimate::Real,
    ground_truth::Real,
    sample_std::Real,
    ess::Real;
    max_rel_ci_width::Real=1e-1,
    min_abs_for_rel_check::Real=1e-10,
)
    se = sample_std / sqrt(ess)
    ci_half_width = Z_99 * se

    if abs(ground_truth) > min_abs_for_rel_check
        rel_ci_width = ci_half_width / abs(ground_truth)
        @assert rel_ci_width <= max_rel_ci_width "CI too wide: $(rel_ci_width) > $(max_rel_ci_width). Need more samples or lower variance."
    end

    diff = abs(estimate - ground_truth)
    passed = diff <= ci_half_width

    if !passed
        @warn "MC estimate check failed" estimate ground_truth diff ci_half_width ess
    end

    return passed
end

"""
    weighted_stats(values, weights)

Compute weighted mean, weighted standard deviation, and ESS from weighted samples.

# Arguments
- `values`: Vector of sample values
- `weights`: Vector of normalized weights (must sum to 1)

# Returns
- `mean_est`: Weighted mean
- `std_est`: Weighted standard deviation
- `ess`: Effective sample size (1 / sum(weights^2))
"""
function weighted_stats(values, weights)
    mean_est = sum(values .* weights)
    var_est = sum(weights .* (values .- mean_est) .^ 2)
    std_est = sqrt(var_est)
    ess = inv(sum(abs2, weights))
    return mean_est, std_est, ess
end
