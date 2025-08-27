@testitem "Batch Kalman test" tags = [:gpu] begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs

    using Random
    using SSMProblems

    using CUDA

    T_elem = Float32  # Use Float32 for type preservation
    rng = StableRNG(1234)
    K = 10
    Dx = 2
    Dy = 2
    μ0s = [rand(rng, T_elem, Dx) for _ in 1:K]
    Σ0s = [rand(rng, T_elem, Dx, Dx) for _ in 1:K]
    Σ0s .= Σ0s .* transpose.(Σ0s)
    As = [rand(rng, T_elem, Dx, Dx) for _ in 1:K]
    bs = [rand(rng, T_elem, Dx) for _ in 1:K]
    Qs = [rand(rng, T_elem, Dx, Dx) for _ in 1:K]
    Qs .= Qs .* transpose.(Qs)
    Hs = [rand(rng, T_elem, Dy, Dx) for _ in 1:K]
    cs = [rand(rng, T_elem, Dy) for _ in 1:K]
    Rs = [rand(rng, T_elem, Dy, Dy) for _ in 1:K]
    Rs .= Rs .* transpose.(Rs)

    models = [
        create_homogeneous_linear_gaussian_model(
            μ0s[k], Σ0s[k], As[k], bs[k], Qs[k], Hs[k], cs[k], Rs[k]
        ) for k in 1:K
    ]

    T = 5
    Ys = [[rand(rng, T_elem, Dy) for _ in 1:T] for _ in 1:K]

    outputs = [
        GeneralisedFilters.filter(rng, models[k], KalmanFilter(), Ys[k]) for k in 1:K
    ]

    states = first.(outputs)
    log_likelihoods = last.(outputs)

    struct BatchGaussianPrior{T,MT} <: GaussianPrior
        μ0s::CuArray{T,2,MT}
        Σ0s::CuArray{T,3,MT}
    end

    function BatchGaussianPrior(μ0s::Vector{Vector{T}}, Σ0s::Vector{Matrix{T}}) where {T}
        μ0s = CuArray(stack(μ0s))
        Σ0s = CuArray(stack(Σ0s))
        return BatchGaussianPrior(μ0s, Σ0s)
    end

    function GeneralisedFilters.batch_calc_μ0s(
        dyn::BatchGaussianPrior, ::Integer; kwargs...
    )
        return dyn.μ0s
    end
    function GeneralisedFilters.batch_calc_Σ0s(
        dyn::BatchGaussianPrior, ::Integer; kwargs...
    )
        return dyn.Σ0s
    end

    struct BatchLinearGaussianDynamics{T,MT} <: LinearGaussianLatentDynamics
        As::CuArray{T,3,MT}
        bs::CuArray{T,2,MT}
        Qs::CuArray{T,3,MT}
    end

    function BatchLinearGaussianDynamics(
        As::Vector{Matrix{T}}, bs::Vector{Vector{T}}, Qs::Vector{Matrix{T}}
    ) where {T}
        As = CuArray(stack(As))
        bs = CuArray(stack(bs))
        Qs = CuArray(stack(Qs))
        return BatchLinearGaussianDynamics(As, bs, Qs)
    end

    function GeneralisedFilters.batch_calc_As(
        dyn::BatchLinearGaussianDynamics, ::Integer, ::Integer; kwargs...
    )
        return dyn.As
    end
    function GeneralisedFilters.batch_calc_bs(
        dyn::BatchLinearGaussianDynamics, ::Integer, ::Integer; kwargs...
    )
        return dyn.bs
    end
    function GeneralisedFilters.batch_calc_Qs(
        dyn::BatchLinearGaussianDynamics, ::Integer, ::Integer; kwargs...
    )
        return dyn.Qs
    end

    struct BatchLinearGaussianObservations{T,MT} <: LinearGaussianObservationProcess
        Hs::CuArray{T,3,MT}
        cs::CuArray{T,2,MT}
        Rs::CuArray{T,3,MT}
    end

    function BatchLinearGaussianObservations(
        Hs::Vector{Matrix{T}}, cs::Vector{Vector{T}}, Rs::Vector{Matrix{T}}
    ) where {T}
        Hs = CuArray(stack(Hs))
        cs = CuArray(stack(cs))
        Rs = CuArray(stack(Rs))
        return BatchLinearGaussianObservations(Hs, cs, Rs)
    end

    function GeneralisedFilters.batch_calc_Hs(
        obs::BatchLinearGaussianObservations, ::Integer, ::Integer; kwargs...
    )
        return obs.Hs
    end
    function GeneralisedFilters.batch_calc_cs(
        obs::BatchLinearGaussianObservations, ::Integer, ::Integer; kwargs...
    )
        return obs.cs
    end
    function GeneralisedFilters.batch_calc_Rs(
        obs::BatchLinearGaussianObservations, ::Integer, ::Integer; kwargs...
    )
        return obs.Rs
    end

    batch_model = GeneralisedFilters.StateSpaceModel(
        BatchGaussianPrior(μ0s, Σ0s),
        BatchLinearGaussianDynamics(As, bs, Qs),
        BatchLinearGaussianObservations(Hs, cs, Rs),
    )

    Ys_batch = Vector{Matrix{T_elem}}(undef, T)
    for t in 1:T
        Ys_batch[t] = stack(Ys[k][t] for k in 1:K)
    end
    batch_output = GeneralisedFilters.filter(
        rng, batch_model, BatchKalmanFilter(K), Ys_batch
    )

    # println("Batch log-likelihood: ", batch_output[2])
    # println("Individual log-likelihoods: ", log_likelihoods)

    # println("Batch states: ", batch_output[1].μs')
    # println("Individual states: ", getproperty.(states, :μ))

    @test Array(batch_output[2])[end] .≈ log_likelihoods[end] rtol = 1e-5
    @test Array(batch_output[1].μs) ≈ stack(getproperty.(states, :μ)) rtol = 1e-5
    # Type preservation tests
    @test eltype(batch_output[1].μs) == T_elem
    @test eltype(batch_output[1].Σs) == T_elem
    @test eltype(batch_output[2]) == T_elem
    @test all(eltype(μ0) == T_elem for μ0 in μ0s)
    @test all(eltype(Σ0) == T_elem for Σ0 in Σ0s)
    @test all(eltype(A) == T_elem for A in As)
    @test all(eltype(b) == T_elem for b in bs)
    @test all(eltype(Q) == T_elem for Q in Qs)
    @test all(eltype(H) == T_elem for H in Hs)
    @test all(eltype(c) == T_elem for c in cs)
    @test all(eltype(R) == T_elem for R in Rs)
    @test all(all(eltype(y) == T_elem for y in Y) for Y in Ys)
    @test all(eltype(Y_batch) == T_elem for Y_batch in Ys_batch)
end
