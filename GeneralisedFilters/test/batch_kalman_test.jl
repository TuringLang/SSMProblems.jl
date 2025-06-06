@testitem "Batch Kalman test" tags = [:gpu] begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs

    using Random
    using SSMProblems

    using CUDA

    rng = StableRNG(1234)
    K = 10
    Dx = 2
    Dy = 2
    μ0s = [rand(rng, Dx) for _ in 1:K]
    Σ0s = [rand(rng, Dx, Dx) for _ in 1:K]
    Σ0s .= Σ0s .* transpose.(Σ0s)
    As = [rand(rng, Dx, Dx) for _ in 1:K]
    bs = [rand(rng, Dx) for _ in 1:K]
    Qs = [rand(rng, Dx, Dx) for _ in 1:K]
    Qs .= Qs .* transpose.(Qs)
    Hs = [rand(rng, Dy, Dx) for _ in 1:K]
    cs = [rand(rng, Dy) for _ in 1:K]
    Rs = [rand(rng, Dy, Dy) for _ in 1:K]
    Rs .= Rs .* transpose.(Rs)

    models = [
        create_homogeneous_linear_gaussian_model(
            μ0s[k], Σ0s[k], As[k], bs[k], Qs[k], Hs[k], cs[k], Rs[k]
        ) for k in 1:K
    ]

    T = 5
    ys_cpu = [rand(rng, Dy) for _ in 1:T]
    Ys = [ys_cpu for _ in 1:K]

    outputs = [
        GeneralisedFilters.filter(rng, models[k], KalmanFilter(), Ys[k]) for k in 1:K
    ]
    states = first.(outputs)
    log_likelihoods = last.(outputs)

    # Define batched model
    μ0 = BatchedCuVector(cu(stack(μ0s)))
    Σ0 = BatchedCuMatrix(cu(stack(Σ0s)))
    A = BatchedCuMatrix(cu(stack(As)))
    b = BatchedCuVector(cu(stack(bs)))
    Q = BatchedCuMatrix(cu(stack(Qs)))

    dyn = GeneralisedFilters.HomogeneousLinearGaussianLatentDynamics(μ0, Σ0, A, b, Q)

    H = BatchedCuMatrix(cu(stack(Hs)))
    c = BatchedCuVector(cu(stack(cs)))
    R = BatchedCuMatrix(cu(stack(Rs)))

    obs = GeneralisedFilters.HomogeneousLinearGaussianObservationProcess(H, c, R)

    ssm = StateSpaceModel(dyn, obs)
    ys = cu.(ys_cpu)

    # Hack: manually setting of initialisation for this model
    function GeneralisedFilters.initialise_log_evidence(
        ::KalmanFilter,
        model::StateSpaceModel{
            T,
            <:GeneralisedFilters.HomogeneousLinearGaussianLatentDynamics{
                T,<:BatchedCuVector
            },
        },
    ) where {T}
        D = size(model.dyn.μ0.data, 2)
        return CUDA.zeros(T, D)
    end

    state, ll = GeneralisedFilters.filter(rng, ssm, KalmanFilter(), ys)

    @test all(isapprox.(Array(ll), log_likelihoods; rtol=1e-5))
    @test Array(state.μ.data) ≈ stack(getproperty.(states, :μ)) rtol = 1e-5
end
