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
struct InnerDynamics{TMat<:AbstractMatrix,TVec<:AbstractVector,TCov<:AbstractMatrix} <:
       LinearGaussianLatentDynamics
    A::TMat
    b::TVec
    C::TMat
    Q::TCov
end

struct InnerPrior{TVec<:AbstractVector,TCov<:AbstractMatrix} <: GaussianPrior
    μ0::TVec
    Σ0::TCov
end

# CPU methods
GeneralisedFilters.calc_μ0(prior::InnerPrior; kwargs...) = prior.μ0
GeneralisedFilters.calc_Σ0(prior::InnerPrior; kwargs...) = prior.Σ0
GeneralisedFilters.calc_A(dyn::InnerDynamics, ::Integer; kwargs...) = dyn.A
function GeneralisedFilters.calc_b(dyn::InnerDynamics, ::Integer; prev_outer, kwargs...)
    return dyn.b + dyn.C * prev_outer
end
GeneralisedFilters.calc_Q(dyn::InnerDynamics, ::Integer; kwargs...) = dyn.Q

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

    # Create full model
    full_model = create_homogeneous_linear_gaussian_model(
        μ0, PDMat(Σ0), A, b, PDMat(Q), H, c, PDMat(R)
    )

    outer_prior, outer_dyn = if static_arrays
        prior = HomogeneousGaussianPrior(
            SVector{D_outer,T}(μ0[1:D_outer]),
            PDMat(SMatrix{D_outer,D_outer,T}(Σ0[1:D_outer, 1:D_outer])),
        )
        dyn = HomogeneousLinearGaussianLatentDynamics(
            SMatrix{D_outer,D_outer,T}(A[1:D_outer, 1:D_outer]),
            SVector{D_outer,T}(b[1:D_outer]),
            PDMat(SMatrix{D_outer,D_outer,T}(Q[1:D_outer, 1:D_outer])),
        )
        prior, dyn
    else
        prior = HomogeneousGaussianPrior(μ0[1:D_outer], PDMat(Σ0[1:D_outer, 1:D_outer]))
        dyn = HomogeneousLinearGaussianLatentDynamics(
            A[1:D_outer, 1:D_outer], b[1:D_outer], PDMat(Q[1:D_outer, 1:D_outer])
        )
        prior, dyn
    end

    inner_prior, inner_dyn = if static_arrays
        prior = InnerPrior(
            SVector{D_inner,T}(μ0[(D_outer + 1):end]),
            PDMat(SMatrix{D_inner,D_inner,T}(Σ0[(D_outer + 1):end, (D_outer + 1):end])),
        )
        dyn = InnerDynamics(
            SMatrix{D_inner,D_outer,T}(A[(D_outer + 1):end, (D_outer + 1):end]),
            SVector{D_inner,T}(b[(D_outer + 1):end]),
            SMatrix{D_inner,D_outer,T}(A[(D_outer + 1):end, 1:D_outer]),
            PDMat(SMatrix{D_inner,D_inner,T}(Q[(D_outer + 1):end, (D_outer + 1):end])),
        )
        prior, dyn
    else
        prior = InnerPrior(
            μ0[(D_outer + 1):end], PDMat(Σ0[(D_outer + 1):end, (D_outer + 1):end])
        )
        dyn = InnerDynamics(
            A[(D_outer + 1):end, (D_outer + 1):end],
            b[(D_outer + 1):end],
            A[(D_outer + 1):end, 1:D_outer],
            PDMat(Q[(D_outer + 1):end, (D_outer + 1):end]),
        )
        prior, dyn
    end

    obs = if static_arrays
        HomogeneousLinearGaussianObservationProcess(
            SMatrix{Dy,D_inner,T}(H[:, (D_outer + 1):end]),
            SVector{Dy,T}(c),
            PDMat(SMatrix{Dy,Dy,T}(R)),
        )
    else
        HomogeneousLinearGaussianObservationProcess(H[:, (D_outer + 1):end], c, PDMat(R))
    end
    hier_model = HierarchicalSSM(outer_prior, outer_dyn, inner_prior, inner_dyn, obs)

    return full_model, hier_model
end
