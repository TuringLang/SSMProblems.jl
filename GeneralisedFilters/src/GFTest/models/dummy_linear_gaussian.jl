"""
    Defines a dummy linear Gaussian model used for testing Rao-Blackwellised algorithms.

    This file defines a (time-homogeneous) linear Gaussian model that can act like a generic
    Rao-Blackwellised model. Splitting the latent dynamics into two components, which we
    call the outer and inner states, with dimensions D_outer and D_inner respectively, the
    joint model is simply a linear Gaussian model with the following restrictions:
    - The upper-right D_outer x D_inner block of the state transition matrix is zero.
    - The transition covariance is block diagonal
    - The observation matrix has the form [0, H] where H is a Dy x D_inner matrix.

    This structure ensures that conditioning on the outer state results in linear Gaussian
    sub-model. By using the implicitly defined simulation methods for the linear Gaussian
    dynamics, this model can be used in Rao-Blackwellised settings.
"""

export InnerDriftFn, create_dummy_linear_gaussian_model, with_inner_drift

"""
    InnerDriftFn(b, C)

Callable used as a [`TimeVarying`](@ref) parameter for the inner-dynamics drift
`b + C * c.prev_outer`. Exposes `b` and `C` as fields so that helpers like
[`with_inner_drift`](@ref) can rebuild the model with a different drift while preserving
`C`.
"""
struct InnerDriftFn{T<:AbstractVector,CT<:AbstractMatrix}
    b::T
    C::CT
end
(f::InnerDriftFn)(t, c) = f.b + f.C * c.prev_outer

function create_dummy_linear_gaussian_model(
    rng::AbstractRNG,
    D_outer::Integer,
    D_inner::Integer,
    Dy::Integer,
    T::Type{<:Real}=Float64;
    static_arrays::Bool=false,
    process_noise_scale=T(0.1),
    obs_noise_scale=T(1.0),
)
    # Generate model matrices/vectors
    μ0 = rand(rng, T, D_outer + D_inner)
    Σ0 = [
        rand_cov(rng, T, D_outer) zeros(T, D_outer, D_inner)
        zeros(T, D_inner, D_outer) rand_cov(rng, T, D_inner)
    ]
    A = [
        rand(rng, T, D_outer, D_outer) zeros(T, D_outer, D_inner)
        rand(rng, T, D_inner, D_outer) rand(rng, T, D_inner, D_inner)
    ]
    b = rand(rng, T, D_outer + D_inner)
    Q11 = rand_cov(rng, T, D_outer; scale=process_noise_scale)
    Q22 = rand_cov(rng, T, D_inner; scale=process_noise_scale)
    Q = [
        Q11 zeros(T, D_outer, D_inner)
        zeros(T, D_inner, D_outer) Q22
    ]
    H = [zeros(T, Dy, D_outer) rand(rng, T, Dy, D_inner)]
    c = rand(rng, T, Dy)
    R = rand_cov(rng, T, Dy; scale=obs_noise_scale)

    full_model = if static_arrays
        create_homogeneous_linear_gaussian_model(
            SVector{D_outer + D_inner,T}(μ0),
            PDMat(SMatrix{D_outer + D_inner,D_outer + D_inner,T}(Σ0)),
            SMatrix{D_outer + D_inner,D_outer + D_inner,T}(A),
            SVector{D_outer + D_inner,T}(b),
            PDMat(SMatrix{D_outer + D_inner,D_outer + D_inner,T}(Q)),
            SMatrix{Dy,D_outer + D_inner,T}(H),
            SVector{Dy,T}(c),
            PDMat(SMatrix{Dy,Dy,T}(R)),
        )
    else
        create_homogeneous_linear_gaussian_model(μ0, PDMat(Σ0), A, b, PDMat(Q), H, c, PDMat(R))
    end

    outer_prior, outer_dyn = if static_arrays
        prior = GaussianPrior(
            SVector{D_outer,T}(μ0[1:D_outer]),
            PDMat(SMatrix{D_outer,D_outer,T}(Σ0[1:D_outer, 1:D_outer])),
        )
        dyn = LinearGaussianLatentDynamics(
            SMatrix{D_outer,D_outer,T}(A[1:D_outer, 1:D_outer]),
            SVector{D_outer,T}(b[1:D_outer]),
            PDMat(SMatrix{D_outer,D_outer,T}(Q[1:D_outer, 1:D_outer])),
        )
        prior, dyn
    else
        prior = GaussianPrior(μ0[1:D_outer], PDMat(Σ0[1:D_outer, 1:D_outer]))
        dyn = LinearGaussianLatentDynamics(
            A[1:D_outer, 1:D_outer], b[1:D_outer], PDMat(Q[1:D_outer, 1:D_outer])
        )
        prior, dyn
    end

    inner_prior, inner_dyn = if static_arrays
        A_in = SMatrix{D_inner,D_inner,T}(A[(D_outer + 1):end, (D_outer + 1):end])
        b_in = SVector{D_inner,T}(b[(D_outer + 1):end])
        C_in = SMatrix{D_inner,D_outer,T}(A[(D_outer + 1):end, 1:D_outer])
        Q_in = PDMat(SMatrix{D_inner,D_inner,T}(Q[(D_outer + 1):end, (D_outer + 1):end]))
        μ0_in = SVector{D_inner,T}(μ0[(D_outer + 1):end])
        Σ0_in = PDMat(SMatrix{D_inner,D_inner,T}(Σ0[(D_outer + 1):end, (D_outer + 1):end]))
        prior = GaussianPrior(μ0_in, Σ0_in)
        dyn = LinearGaussianLatentDynamics(
            A_in, TimeVarying(InnerDriftFn(b_in, C_in)), Q_in
        )
        prior, dyn
    else
        A_in = A[(D_outer + 1):end, (D_outer + 1):end]
        b_in = b[(D_outer + 1):end]
        C_in = A[(D_outer + 1):end, 1:D_outer]
        Q_in = PDMat(Q[(D_outer + 1):end, (D_outer + 1):end])
        prior = GaussianPrior(
            μ0[(D_outer + 1):end], PDMat(Σ0[(D_outer + 1):end, (D_outer + 1):end])
        )
        dyn = LinearGaussianLatentDynamics(
            A_in, TimeVarying(InnerDriftFn(b_in, C_in)), Q_in
        )
        prior, dyn
    end

    obs = if static_arrays
        LinearGaussianObservationProcess(
            SMatrix{Dy,D_inner,T}(H[:, (D_outer + 1):end]),
            SVector{Dy,T}(c),
            PDMat(SMatrix{Dy,Dy,T}(R)),
        )
    else
        LinearGaussianObservationProcess(H[:, (D_outer + 1):end], c, PDMat(R))
    end
    hier_model = HierarchicalSSM(outer_prior, outer_dyn, inner_prior, inner_dyn, obs)

    return full_model, hier_model
end

"""
    with_inner_drift(model::HierarchicalSSM, b)

Return a copy of a dummy linear Gaussian `HierarchicalSSM` with the inner-dynamics drift
constant replaced by `b`. Preserves the original `C` matrix from [`InnerDriftFn`](@ref).
"""
function with_inner_drift(model::HierarchicalSSM, b::AbstractVector)
    inner_dyn = model.inner_model.dyn
    drift = inner_dyn.b.f
    b_typed = _convert_like(b, drift.b)
    new_drift = InnerDriftFn(b_typed, drift.C)
    new_inner_dyn = LinearGaussianLatentDynamics(
        inner_dyn.A, TimeVarying(new_drift), inner_dyn.Q
    )
    return HierarchicalSSM(
        model.outer_prior,
        model.outer_dyn,
        model.inner_model.prior,
        new_inner_dyn,
        model.inner_model.obs,
    )
end

function _convert_like(x::AbstractVector, template::StaticArrays.StaticVector{N}) where {N}
    return SVector{N}(x)
end

_convert_like(x::AbstractVector, ::AbstractVector) = x
