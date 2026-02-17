using StaticArrays
import PDMats: PDMat

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

    return create_homogeneous_linear_gaussian_model(
        μ0, PDMat(Σ0), A, b, PDMat(Q), H, c, PDMat(R)
    )
end

## NON-HOMOGENEOUS LINEAR GAUSSIAN MODEL FOR TESTING ####

struct NonHomogeneousLinearGaussianLatentDynamics{
    AT<:AbstractVector,bT<:AbstractVector,QT<:AbstractVector
} <: GeneralisedFilters.LinearGaussianLatentDynamics
    As::AT
    bs::bT
    Qs::QT
end

function GeneralisedFilters.calc_A(
    dyn::NonHomogeneousLinearGaussianLatentDynamics, step::Integer; kwargs...
)
    return dyn.As[step]
end
function GeneralisedFilters.calc_b(
    dyn::NonHomogeneousLinearGaussianLatentDynamics, step::Integer; kwargs...
)
    return dyn.bs[step]
end
function GeneralisedFilters.calc_Q(
    dyn::NonHomogeneousLinearGaussianLatentDynamics, step::Integer; kwargs...
)
    return dyn.Qs[step]
end

struct NonHomogeneousLinearGaussianObservationProcess{
    HT<:AbstractVector,cT<:AbstractVector,RT<:AbstractVector
} <: GeneralisedFilters.LinearGaussianObservationProcess
    Hs::HT
    cs::cT
    Rs::RT
end

function GeneralisedFilters.calc_H(
    obs::NonHomogeneousLinearGaussianObservationProcess, step::Integer; kwargs...
)
    return obs.Hs[step]
end
function GeneralisedFilters.calc_c(
    obs::NonHomogeneousLinearGaussianObservationProcess, step::Integer; kwargs...
)
    return obs.cs[step]
end
function GeneralisedFilters.calc_R(
    obs::NonHomogeneousLinearGaussianObservationProcess, step::Integer; kwargs...
)
    return obs.Rs[step]
end

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
    prior = GeneralisedFilters.HomogeneousGaussianPrior(μ0, PDMat(Σ0))

    As = [rand(rng, T, Dx, Dx) for _ in 1:T_max]
    bs = [rand(rng, T, Dx) for _ in 1:T_max]
    Qs = [PDMat(rand_cov(rng, T, Dx; scale=process_noise_scale)) for _ in 1:T_max]
    dyn = NonHomogeneousLinearGaussianLatentDynamics(As, bs, Qs)

    Hs = [rand(rng, T, Dy, Dx) for _ in 1:T_max]
    cs = [rand(rng, T, Dy) for _ in 1:T_max]
    Rs = [PDMat(rand_cov(rng, T, Dy; scale=obs_noise_scale)) for _ in 1:T_max]
    obs = NonHomogeneousLinearGaussianObservationProcess(Hs, cs, Rs)

    return SSMProblems.StateSpaceModel(prior, dyn, obs)
end

function _compute_joint_nonhomogeneous(model, T::Integer)
    (; μ0, Σ0) = model.prior
    dyn = model.dyn
    obs = model.obs
    Dy, Dx = size(calc_H(obs, 1))

    # Let Z = [X0, X1, ..., XT, Y1, ..., YT] be the joint state vector
    # Write Z = P.Z + ϵ, where ϵ ~ N(μ_ϵ, Σ_ϵ)
    P = zeros(Dx + T * (Dx + Dy), Dx + T * (Dx + Dy))
    for t in 1:T
        A_t = calc_A(dyn, t)
        H_t = calc_H(obs, t)

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
        b_t = calc_b(dyn, t)
        c_t = calc_c(obs, t)

        ib = t * Dx + 1
        μ_ϵ[ib:(ib + Dx - 1)] = b_t

        ic = Dx * (T + 1) + (t - 1) * Dy + 1
        μ_ϵ[ic:(ic + Dy - 1)] = c_t
    end

    Σ_ϵ = zeros(Dx + T * (Dx + Dy), Dx + T * (Dx + Dy))
    Σ_ϵ[1:Dx, 1:Dx] .= Σ0
    for t in 1:T
        Q_t = calc_Q(dyn, t)
        R_t = calc_R(obs, t)

        iQ = t * Dx + 1
        Σ_ϵ[iQ:(iQ + Dx - 1), iQ:(iQ + Dx - 1)] = Q_t

        iR = Dx * (T + 1) + (t - 1) * Dy + 1
        Σ_ϵ[iR:(iR + Dy - 1), iR:(iR + Dy - 1)] = R_t
    end

    # Note (I - P)Z = ϵ and solve for Z ~ N(μ_Z, Σ_Z)
    μ_Z = (I - P) \ μ_ϵ
    Σ_Z = ((I - P) \ Σ_ϵ) / (I - P)'

    return μ_Z, Σ_Z
end

function _compute_joint(model, T::Integer)
    (; μ0, Σ0) = model.prior
    (; A, b, Q) = model.dyn
    (; H, c, R) = model.obs
    Dy, Dx = size(H)

    # Let Z = [X0, X1, ..., XT, Y1, ..., YT] be the joint state vector
    # Write Z = P.Z + ϵ, where ϵ ~ N(μ_ϵ, Σ_ϵ)
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

    # Note (I - P)Z = ϵ and solve for Z ~ N(μ_Z, Σ_Z)
    μ_Z = (I - P) \ μ_ϵ
    Σ_Z = ((I - P) \ Σ_ϵ) / (I - P)'

    return μ_Z, Σ_Z
end

## GRADIENT TEST HELPERS ##

"""
    setup_gradient_test(rng; D=2, T=3)

Set up a gradient test scenario with a linear Gaussian model.
Returns a named tuple with all quantities needed for gradient testing.
"""
function setup_gradient_test(rng::AbstractRNG; D::Int=2, T::Int=3)
    model = create_linear_gaussian_model(rng, D, D, Float64, 0.1, 1.0; static_arrays=true)
    _, _, ys = SSMProblems.sample(rng, model, T)

    A = GeneralisedFilters.calc_A(SSMProblems.dyn(model), 1)
    b = GeneralisedFilters.calc_b(SSMProblems.dyn(model), 1)
    Q = GeneralisedFilters.calc_Q(SSMProblems.dyn(model), 1)
    H, c, R = GeneralisedFilters.calc_params(SSMProblems.obs(model), 1)
    μ0 = GeneralisedFilters.calc_μ0(SSMProblems.prior(model))
    Σ0 = GeneralisedFilters.calc_Σ0(SSMProblems.prior(model))

    state = GeneralisedFilters.initialise(
        rng, SSMProblems.prior(model), GeneralisedFilters.KF()
    )
    caches = Vector{GeneralisedFilters.KalmanGradientCache}(undef, T)
    μ_prevs = Vector{typeof(state.μ)}(undef, T)
    Σ_prevs = Vector{typeof(state.Σ)}(undef, T)

    for t in 1:T
        μ_prevs[t] = state.μ
        Σ_prevs[t] = state.Σ
        pred = GeneralisedFilters.predict(
            rng, SSMProblems.dyn(model), GeneralisedFilters.KF(), t, state, ys[t]
        )
        state, _, caches[t] = GeneralisedFilters.update_with_cache(
            SSMProblems.obs(model), GeneralisedFilters.KF(), t, pred, ys[t]
        )
    end

    ∂μ = @SVector zeros(D)
    ∂Σ = @SMatrix zeros(D, D)
    ∂Q_total = @SMatrix zeros(D, D)
    ∂R_total = @SMatrix zeros(D, D)
    ∂A_total = @SMatrix zeros(D, D)
    ∂b_total = @SVector zeros(D)
    ∂H_total = @SMatrix zeros(D, D)
    ∂c_total = @SVector zeros(D)

    for t in T:-1:1
        ∂μ_pred, ∂Σ_pred = GeneralisedFilters.backward_gradient_update(
            ∂μ, ∂Σ, caches[t], H, R
        )
        ∂Q_total += GeneralisedFilters.gradient_Q(∂Σ_pred)
        ∂R_total += GeneralisedFilters.gradient_R(∂μ, ∂Σ, caches[t])
        ∂A_total += GeneralisedFilters.gradient_A(
            ∂μ_pred, ∂Σ_pred, μ_prevs[t], Σ_prevs[t], A
        )
        ∂b_total += GeneralisedFilters.gradient_b(∂μ_pred)
        ∂H_total += GeneralisedFilters.gradient_H(∂μ, ∂Σ, caches[t], caches[t].Σ_pred, H)
        ∂c_total += GeneralisedFilters.gradient_c(∂μ, caches[t])
        ∂μ, ∂Σ = GeneralisedFilters.backward_gradient_predict(∂μ_pred, ∂Σ_pred, A)
    end

    ∂μ0 = ∂μ
    ∂Σ0 = ∂Σ

    return (;
        D,
        T,
        model,
        ys,
        A,
        b,
        Q,
        H,
        c,
        R,
        μ0,
        Σ0,
        caches,
        μ_prevs,
        Σ_prevs,
        ∂Q_total,
        ∂R_total,
        ∂A_total,
        ∂b_total,
        ∂H_total,
        ∂c_total,
        ∂μ0,
        ∂Σ0,
    )
end

"""
    make_nll_func(model, ys, param::Symbol)

Create a function that computes NLL with one parameter varied.
Returns a function taking a vector and returning scalar NLL.
"""
function make_nll_func(model, ys, param::Symbol)
    pr = SSMProblems.prior(model)
    dy = SSMProblems.dyn(model)
    ob = SSMProblems.obs(model)

    μ0 = GeneralisedFilters.calc_μ0(pr)
    Σ0 = GeneralisedFilters.calc_Σ0(pr)
    A = GeneralisedFilters.calc_A(dy, 1)
    b = GeneralisedFilters.calc_b(dy, 1)
    Q = GeneralisedFilters.calc_Q(dy, 1)
    H = GeneralisedFilters.calc_H(ob, 1)
    c = GeneralisedFilters.calc_c(ob, 1)
    R = GeneralisedFilters.calc_R(ob, 1)

    Dx = length(μ0)
    Dy = length(c)

    function make_pd(M)
        M_sym = (M + M') / 2
        λ, V = eigen(M_sym)
        λ_clipped = max.(λ, 1e-8)
        return PDMat(Symmetric(V * Diagonal(λ_clipped) * V'))
    end

    if param == :Q
        return function (x)
            Q_new = make_pd(reshape(x, Dx, Dx))
            m = GeneralisedFilters.create_homogeneous_linear_gaussian_model(
                μ0, Σ0, A, b, Q_new, H, c, R
            )
            _, ll = GeneralisedFilters.filter(m, GeneralisedFilters.KF(), ys)
            return -ll
        end
    elseif param == :R
        return function (x)
            R_new = make_pd(reshape(x, Dy, Dy))
            m = GeneralisedFilters.create_homogeneous_linear_gaussian_model(
                μ0, Σ0, A, b, Q, H, c, R_new
            )
            _, ll = GeneralisedFilters.filter(m, GeneralisedFilters.KF(), ys)
            return -ll
        end
    elseif param == :A
        return function (x)
            A_new = reshape(x, Dx, Dx)
            m = GeneralisedFilters.create_homogeneous_linear_gaussian_model(
                μ0, Σ0, A_new, b, Q, H, c, R
            )
            _, ll = GeneralisedFilters.filter(m, GeneralisedFilters.KF(), ys)
            return -ll
        end
    elseif param == :b
        return function (x)
            m = GeneralisedFilters.create_homogeneous_linear_gaussian_model(
                μ0, Σ0, A, x, Q, H, c, R
            )
            _, ll = GeneralisedFilters.filter(m, GeneralisedFilters.KF(), ys)
            return -ll
        end
    elseif param == :H
        return function (x)
            H_new = reshape(x, Dy, Dx)
            m = GeneralisedFilters.create_homogeneous_linear_gaussian_model(
                μ0, Σ0, A, b, Q, H_new, c, R
            )
            _, ll = GeneralisedFilters.filter(m, GeneralisedFilters.KF(), ys)
            return -ll
        end
    elseif param == :c
        return function (x)
            m = GeneralisedFilters.create_homogeneous_linear_gaussian_model(
                μ0, Σ0, A, b, Q, H, x, R
            )
            _, ll = GeneralisedFilters.filter(m, GeneralisedFilters.KF(), ys)
            return -ll
        end
    elseif param == :μ0
        return function (x)
            m = GeneralisedFilters.create_homogeneous_linear_gaussian_model(
                x, Σ0, A, b, Q, H, c, R
            )
            _, ll = GeneralisedFilters.filter(m, GeneralisedFilters.KF(), ys)
            return -ll
        end
    elseif param == :Σ0
        return function (x)
            Σ0_new = make_pd(reshape(x, Dx, Dx))
            m = GeneralisedFilters.create_homogeneous_linear_gaussian_model(
                μ0, Σ0_new, A, b, Q, H, c, R
            )
            _, ll = GeneralisedFilters.filter(m, GeneralisedFilters.KF(), ys)
            return -ll
        end
    else
        error("Unknown parameter: $param")
    end
end
