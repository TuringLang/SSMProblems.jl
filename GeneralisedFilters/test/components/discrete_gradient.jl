"""Gradient computation tests for discrete/HMM filter parameters."""

@testitem "Discrete gradient: ∂Ψ (transition logits)" begin
    using FiniteDifferences
    using GeneralisedFilters
    using StableRNGs

    rng = StableRNG(1234)
    s = GeneralisedFilters.GFTest.setup_discrete_gradient_test(rng)
    fdm = central_fdm(5, 1)

    nll_Ψ = GeneralisedFilters.GFTest.make_discrete_nll_func(s.model, s.ys, :Ψ)
    Ψ_vec = vec(s.Ψ)
    ∂Ψ_fd = reshape(FiniteDifferences.grad(fdm, nll_Ψ, Ψ_vec)[1], s.K, s.K)

    @test s.∂Ψ_total ≈ ∂Ψ_fd rtol = 1e-4
end

@testitem "Discrete gradient: ∂μ (emission means)" begin
    using FiniteDifferences
    using GeneralisedFilters
    using StableRNGs

    rng = StableRNG(1234)
    s = GeneralisedFilters.GFTest.setup_discrete_gradient_test(rng)
    fdm = central_fdm(5, 1)

    nll_μ = GeneralisedFilters.GFTest.make_discrete_nll_func(s.model, s.ys, :μ)
    ∂μ_fd = FiniteDifferences.grad(fdm, nll_μ, s.μs)[1]

    @test s.∂μ_total ≈ ∂μ_fd rtol = 1e-4
end

@testitem "Discrete gradient: ∂η0 (initial logits)" begin
    using FiniteDifferences
    using GeneralisedFilters
    using StableRNGs

    rng = StableRNG(1234)
    s = GeneralisedFilters.GFTest.setup_discrete_gradient_test(rng)
    fdm = central_fdm(5, 1)

    nll_η0 = GeneralisedFilters.GFTest.make_discrete_nll_func(s.model, s.ys, :η0)
    ∂η0_fd = FiniteDifferences.grad(fdm, nll_η0, s.η0)[1]

    @test s.∂η0 ≈ ∂η0_fd rtol = 1e-4
end
