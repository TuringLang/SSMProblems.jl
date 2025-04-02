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

export InnerDynamics, create_dummy_linear_gaussian_model

"""
    Inner dynamics of the dummy linear Gaussian model.

    Linear Gaussian dynamics conditonal on the (previous) outer state (u_t), defined by:
    x_{t+1} = A x_t + b + C u_t + w_t
"""
struct InnerDynamics{
    T,TMat<:AbstractMatrix{T},TVec<:AbstractVector{T},TCov<:AbstractMatrix{T}
} <: LinearGaussianLatentDynamics{T}
    μ0::TVec
    Σ0::TCov
    A::TMat
    b::TVec
    C::TMat
    Q::TCov
end

# CPU methods
GeneralisedFilters.calc_μ0(dyn::InnerDynamics; kwargs...) = dyn.μ0
GeneralisedFilters.calc_Σ0(dyn::InnerDynamics; kwargs...) = dyn.Σ0
GeneralisedFilters.calc_A(dyn::InnerDynamics, ::Integer; kwargs...) = dyn.A
function GeneralisedFilters.calc_b(dyn::InnerDynamics, ::Integer; prev_outer, kwargs...)
    return dyn.b + dyn.C * prev_outer
end
GeneralisedFilters.calc_Q(dyn::InnerDynamics, ::Integer; kwargs...) = dyn.Q

# GPU methods
function GeneralisedFilters.batch_calc_μ0s(dyn::InnerDynamics{T}, N; kwargs...) where {T}
    μ0s = CuArray{T}(undef, length(dyn.μ0), N)
    return μ0s[:, :] .= cu(dyn.μ0)
end

function GeneralisedFilters.batch_calc_Σ0s(
    dyn::InnerDynamics{T}, N::Integer; kwargs...
) where {T}
    Σ0s = CuArray{T}(undef, size(dyn.Σ0)..., N)
    return Σ0s[:, :, :] .= cu(dyn.Σ0)
end

function GeneralisedFilters.batch_calc_As(
    dyn::InnerDynamics{T}, ::Integer, N::Integer; kwargs...
) where {T}
    As = CuArray{T}(undef, size(dyn.A)..., N)
    As[:, :, :] .= cu(dyn.A)
    return As
end

function GeneralisedFilters.batch_calc_bs(
    dyn::InnerDynamics{T}, ::Integer, N::Integer; prev_outer, kwargs...
) where {T}
    Cs = CuArray{T}(undef, size(dyn.C)..., N)
    Cs[:, :, :] .= cu(dyn.C)
    return NNlib.batched_vec(Cs, prev_outer) .+ cu(dyn.b)
end

function GeneralisedFilters.batch_calc_Qs(
    dyn::InnerDynamics{T}, ::Integer, N::Integer; kwargs...
) where {T}
    Q = CuArray{T}(undef, size(dyn.Q)..., N)
    return Q[:, :, :] .= cu(dyn.Q)
end

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
    Σ0s = [
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

    # Create full model
    full_model = create_homogeneous_linear_gaussian_model(μ0, Σ0s, A, b, Q, H, c, R)

    # Create hierarchical model
    outer_dyn = GeneralisedFilters.HomogeneousLinearGaussianLatentDynamics(
        μ0[1:D_outer],
        Σ0s[1:D_outer, 1:D_outer],
        A[1:D_outer, 1:D_outer],
        b[1:D_outer],
        Q[1:D_outer, 1:D_outer],
    )
    inner_dyn = if static_arrays
        InnerDynamics(
            SVector{D_inner,T}(μ0[(D_outer + 1):end]),
            SMatrix{D_inner,D_inner,T}(Σ0s[(D_outer + 1):end, (D_outer + 1):end]),
            SMatrix{D_inner,D_outer,T}(A[(D_outer + 1):end, (D_outer + 1):end]),
            SVector{D_inner,T}(b[(D_outer + 1):end]),
            SMatrix{D_inner,D_outer,T}(A[(D_outer + 1):end, 1:D_outer]),
            SMatrix{D_inner,D_inner,T}(Q[(D_outer + 1):end, (D_outer + 1):end]),
        )
    else
        InnerDynamics(
            μ0[(D_outer + 1):end],
            Σ0s[(D_outer + 1):end, (D_outer + 1):end],
            A[(D_outer + 1):end, (D_outer + 1):end],
            b[(D_outer + 1):end],
            A[(D_outer + 1):end, 1:D_outer],
            Q[(D_outer + 1):end, (D_outer + 1):end],
        )
    end
    obs = GeneralisedFilters.HomogeneousLinearGaussianObservationProcess(
        H[:, (D_outer + 1):end], c, R
    )
    hier_model = HierarchicalSSM(outer_dyn, inner_dyn, obs)

    return full_model, hier_model
end
