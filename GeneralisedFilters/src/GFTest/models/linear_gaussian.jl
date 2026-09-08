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

# Exact Gaussian reference for parameter-posterior integration tests.
function augment_drift_model(
    model,
    drift_indices;
    σ²_b::Union{Real,AbstractMatrix}=4.0,
    μ_b::Union{Nothing,AbstractVector}=nothing,
    ε::Real=1e-12,
    static_arrays::Union{Nothing,Bool}=nothing,
)
    pr = model.prior
    dy = model.dyn
    ob = model.obs

    μ0_raw = pr.μ0
    Σ0_raw = pr.Σ0
    A_raw = dy.A
    b_raw = dy.b
    Q_raw = dy.Q
    H_raw = ob.H
    c_raw = ob.c
    R_raw = ob.R

    σ²_b_eltype = σ²_b isa Real ? typeof(σ²_b) : eltype(σ²_b)
    μ_b_eltype = isnothing(μ_b) ? eltype(b_raw) : eltype(μ_b)
    T = promote_type(
        eltype(μ0_raw),
        eltype(Σ0_raw),
        eltype(A_raw),
        eltype(b_raw),
        eltype(Q_raw),
        eltype(H_raw),
        eltype(c_raw),
        eltype(R_raw),
        σ²_b_eltype,
        μ_b_eltype,
        typeof(ε),
    )

    μ0 = Vector{T}(μ0_raw)
    Σ0 = Matrix{T}(Σ0_raw)
    A = Matrix{T}(A_raw)
    b = Vector{T}(b_raw)
    Q = Matrix{T}(Q_raw)
    H = Matrix{T}(H_raw)
    c = Vector{T}(c_raw)
    R = Matrix{T}(R_raw)

    Dx = length(μ0)
    Dy = length(c)
    idx = _collect_drift_indices(drift_indices)
    _validate_drift_indices(idx, Dx)
    K = length(idx)

    μ_b_vec = if isnothing(μ_b)
        zeros(T, K)
    else
        μ_vec = Vector{T}(μ_b)
        length(μ_vec) == K || throw(ArgumentError("μ_b must have length $K."))
        μ_vec
    end
    Σ_b = _drift_prior_covariance(σ²_b, K, T)

    b_fixed = copy(b)
    b_fixed[idx] .= zero(T)

    A_aug = zeros(T, Dx + K, Dx + K)
    A_aug[1:Dx, 1:Dx] = A
    for (j, i) in enumerate(idx)
        A_aug[i, Dx + j] = one(T)
    end
    @inbounds for j in 1:K
        A_aug[Dx + j, Dx + j] = one(T)
    end

    b_aug = vcat(b_fixed, zeros(T, K))

    Q_aug = zeros(T, Dx + K, Dx + K)
    Q_aug[1:Dx, 1:Dx] = Q
    @inbounds for j in 1:K
        Q_aug[Dx + j, Dx + j] = T(ε)
    end

    H_aug = zeros(T, Dy, Dx + K)
    H_aug[:, 1:Dx] = H

    μ0_aug = vcat(μ0, μ_b_vec)
    Σ0_aug = zeros(T, Dx + K, Dx + K)
    Σ0_aug[1:Dx, 1:Dx] = Σ0
    Σ0_aug[(Dx + 1):end, (Dx + 1):end] = Σ_b

    use_static = isnothing(static_arrays) ? _has_static_lg_arrays(model) : static_arrays

    μ0_out = _maybe_static_vector(μ0_aug, use_static)
    A_out = _maybe_static_matrix(A_aug, use_static)
    b_out = _maybe_static_vector(b_aug, use_static)
    H_out = _maybe_static_matrix(H_aug, use_static)
    c_out = _maybe_static_vector(c, use_static)
    Σ0_out = _maybe_static_matrix((Σ0_aug + Σ0_aug') / 2, use_static)
    Q_out = _maybe_static_matrix((Q_aug + Q_aug') / 2, use_static)
    R_out = _maybe_static_matrix((R + R') / 2, use_static)

    aug_model = create_homogeneous_linear_gaussian_model(
        μ0_out, Σ0_out, A_out, b_out, Q_out, H_out, c_out, R_out
    )

    return (model=aug_model, drift_slice=(Dx + 1):(Dx + K), drift_indices=idx)
end

"""
    augmented_kf_drift_posterior(model, observations, drift_indices; kwargs...)

Run a Kalman filter on the augmented model from `augment_drift_model` and return
posterior mean/std for unknown drift components.
"""
function augmented_kf_drift_posterior(model, observations, drift_indices; kwargs...)
    aug = augment_drift_model(model, drift_indices; kwargs...)
    state, ll = GeneralisedFilters.filter(aug.model, GeneralisedFilters.KF(), observations)
    Σ = Matrix(state.Σ)
    μ_post = state.μ[aug.drift_slice]
    σ_post = sqrt.(diag(Σ)[aug.drift_slice])
    return (;
        state,
        log_likelihood=ll,
        mean=μ_post,
        std=σ_post,
        augmented_model=aug.model,
        drift_slice=aug.drift_slice,
    )
end

function _collect_drift_indices(drift_indices::Integer)
    return [Int(drift_indices)]
end

function _collect_drift_indices(drift_indices)
    return collect(Int, drift_indices)
end

function _validate_drift_indices(idx::AbstractVector{<:Integer}, Dx::Integer)
    isempty(idx) && throw(ArgumentError("drift_indices cannot be empty."))
    any(i -> i < 1 || i > Dx, idx) &&
        throw(ArgumentError("drift_indices must be between 1 and $Dx."))
    length(unique(idx)) == length(idx) ||
        throw(ArgumentError("drift_indices must not contain duplicates."))
    return nothing
end

function _drift_prior_covariance(σ²_b::Real, K::Integer, ::Type{T}) where {T}
    Σ = zeros(T, K, K)
    @inbounds for i in 1:K
        Σ[i, i] = T(σ²_b)
    end
    return Σ
end

function _drift_prior_covariance(σ²_b::AbstractMatrix, K::Integer, ::Type{T}) where {T}
    size(σ²_b) == (K, K) || throw(ArgumentError("σ²_b matrix must have size ($K, $K)."))
    return Matrix{T}(σ²_b)
end

function _has_static_lg_arrays(model)
    return model.prior.μ0 isa StaticArray ||
           model.dyn.A isa StaticArray ||
           model.obs.H isa StaticArray
end

function _maybe_static_vector(x::AbstractVector, static_arrays::Bool)
    if static_arrays
        return SVector{length(x),eltype(x)}(x)
    end
    return x
end

function _maybe_static_matrix(X::AbstractMatrix, static_arrays::Bool)
    if static_arrays
        nr, nc = size(X)
        return SMatrix{nr,nc,eltype(X)}(X)
    end
    return X
end
