using StaticArrays

function create_linear_gaussian_model(
    rng::AbstractRNG,
    Dx::Integer,
    Dy::Integer,
    T::Type{<:Real}=Float64,
    process_noise_scale=T(0.1),
    obs_noise_scale=T(1.0);
    static_arrays::Bool=false,
)
    μ0 = rand(rng, T, Dx)
    Σ0 = rand_cov(rng, T, Dx)
    A = rand(rng, T, Dx, Dx)
    b = rand(rng, T, Dx)
    Q = rand_cov(rng, T, Dx; scale=process_noise_scale)
    H = rand(rng, T, Dy, Dx)
    c = rand(rng, T, Dy)
    R = rand_cov(rng, T, Dy; scale=obs_noise_scale)

    if static_arrays
        μ0 = SVector{Dx,T}(μ0)
        Σ0 = SMatrix{Dx,Dx,T}(Σ0)
        A = SMatrix{Dx,Dx,T}(A)
        b = SVector{Dx,T}(b)
        Q = SMatrix{Dx,Dx,T}(Q)
        H = SMatrix{Dy,Dx,T}(H)
        c = SVector{Dy,T}(c)
        R = SMatrix{Dy,Dy,T}(R)
    end

    return create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
end

## NON-HOMOGENEOUS LINEAR GAUSSIAN MODEL FOR TESTING #######################################

function create_nonhomogeneous_linear_gaussian_model(
    rng::AbstractRNG,
    Dx::Integer,
    Dy::Integer,
    T_max::Integer,
    (::Type{T})=Float64,
    process_noise_scale=T(0.1),
    obs_noise_scale=T(1.0),
) where {T<:Real}
    μ0 = rand(rng, T, Dx)
    Σ0 = rand_cov(rng, T, Dx)
    prior = GaussianPrior(μ0, Σ0)

    As = [rand(rng, T, Dx, Dx) for _ in 1:T_max]
    bs = [rand(rng, T, Dx) for _ in 1:T_max]
    Qs = [rand_cov(rng, T, Dx; scale=process_noise_scale) for _ in 1:T_max]
    dyn = TimeVaryingDynamics(((; t),) -> LinearGaussianDynamics(As[t], bs[t], Qs[t]))

    Hs = [rand(rng, T, Dy, Dx) for _ in 1:T_max]
    cs = [rand(rng, T, Dy) for _ in 1:T_max]
    Rs = [rand_cov(rng, T, Dy; scale=obs_noise_scale) for _ in 1:T_max]
    obs = TimeVaryingObservation(((; t),) -> LinearGaussianObservation(Hs[t], cs[t], Rs[t]))

    return StateSpaceModel(prior, dyn, obs)
end

## JOINT DISTRIBUTIONS FOR ANALYTIC COMPARISON ############################################

# Both helpers build Z = [X0, X1, ..., XT, Y1, ..., YT], write Z = P Z + ϵ with
# ϵ ~ N(μ_ϵ, Σ_ϵ), and solve (I - P) Z = ϵ for the joint moments (μ_Z, Σ_Z).

function _compute_joint_nonhomogeneous(model, T::Integer)
    (; μ0, Σ0) = model.prior
    d(t) = GeneralisedFilters.resolve(model.dyn, (; t))
    o(t) = GeneralisedFilters.resolve(model.obs, (; t))
    Dy, Dx = size(o(1).H)

    P = zeros(Dx + T * (Dx + Dy), Dx + T * (Dx + Dy))
    for t in 1:T
        A_t = d(t).A
        H_t = o(t).H

        iA = t * Dx + 1
        jA = (t - 1) * Dx + 1
        P[iA:(iA + Dx - 1), jA:(jA + Dx - 1)] = A_t

        iH = Dx * (T + 1) + (t - 1) * Dy + 1
        jH = Dx * t + 1
        P[iH:(iH + Dy - 1), jH:(jH + Dx - 1)] = H_t
    end

    μ_ϵ = zeros(Dx + T * (Dx + Dy))
    μ_ϵ[1:Dx] .= μ0
    for t in 1:T
        b_t = d(t).b
        c_t = o(t).c

        ib = t * Dx + 1
        μ_ϵ[ib:(ib + Dx - 1)] = b_t

        ic = Dx * (T + 1) + (t - 1) * Dy + 1
        μ_ϵ[ic:(ic + Dy - 1)] = c_t
    end

    Σ_ϵ = zeros(Dx + T * (Dx + Dy), Dx + T * (Dx + Dy))
    Σ_ϵ[1:Dx, 1:Dx] .= Σ0
    for t in 1:T
        Q_t = d(t).Q
        R_t = o(t).R

        iQ = t * Dx + 1
        Σ_ϵ[iQ:(iQ + Dx - 1), iQ:(iQ + Dx - 1)] = Q_t

        iR = Dx * (T + 1) + (t - 1) * Dy + 1
        Σ_ϵ[iR:(iR + Dy - 1), iR:(iR + Dy - 1)] = R_t
    end

    μ_Z = (I - P) \ μ_ϵ
    Σ_Z = ((I - P) \ Σ_ϵ) / (I - P)'

    return μ_Z, Σ_Z
end

function _compute_joint(model, T::Integer)
    (; μ0, Σ0) = model.prior
    (; A, b, Q) = model.dyn
    (; H, c, R) = model.obs
    Dy, Dx = size(H)

    P = zeros(Dx + T * (Dx + Dy), Dx + T * (Dx + Dy))
    for t in 1:T
        iA = t * Dx + 1
        jA = (t - 1) * Dx + 1
        P[iA:(iA + Dx - 1), jA:(jA + Dx - 1)] = A

        iH = Dx * (T + 1) + (t - 1) * Dy + 1
        jH = Dx * t + 1
        P[iH:(iH + Dy - 1), jH:(jH + Dx - 1)] = H
    end

    μ_ϵ = zeros(Dx + T * (Dx + Dy))
    μ_ϵ[1:Dx] .= μ0
    for t in 1:T
        ib = t * Dx + 1
        μ_ϵ[ib:(ib + Dx - 1)] = b

        ic = Dx * (T + 1) + (t - 1) * Dy + 1
        μ_ϵ[ic:(ic + Dy - 1)] = c
    end

    Σ_ϵ = zeros(Dx + T * (Dx + Dy), Dx + T * (Dx + Dy))
    Σ_ϵ[1:Dx, 1:Dx] .= Σ0
    for t in 1:T
        iQ = t * Dx + 1
        Σ_ϵ[iQ:(iQ + Dx - 1), iQ:(iQ + Dx - 1)] = Q

        iR = Dx * (T + 1) + (t - 1) * Dy + 1
        Σ_ϵ[iR:(iR + Dy - 1), iR:(iR + Dy - 1)] = R
    end

    μ_Z = (I - P) \ μ_ϵ
    Σ_Z = ((I - P) \ Σ_ϵ) / (I - P)'

    return μ_Z, Σ_Z
end
