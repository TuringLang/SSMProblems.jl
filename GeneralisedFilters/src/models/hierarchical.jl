import SSMProblems: LatentDynamics, ObservationProcess, simulate
export HierarchicalSSM

struct HierarchicalSSM{T<:Real,OD<:LatentDynamics{T},M<:StateSpaceModel{T}} <:
       AbstractStateSpaceModel
    outer_dyn::OD
    inner_model::M
    function HierarchicalSSM(
        outer_dyn::LatentDynamics{T}, inner_model::StateSpaceModel{T}
    ) where {T}
        return new{T,typeof(outer_dyn),typeof(inner_model)}(outer_dyn, inner_model)
    end
end

function HierarchicalSSM(
    outer_dyn::LatentDynamics{T}, inner_dyn::LatentDynamics{T}, obs::ObservationProcess{T}
) where {T}
    inner_model = StateSpaceModel(inner_dyn, obs)
    return HierarchicalSSM(outer_dyn, inner_model)
end

SSMProblems.arithmetic_type(::Type{<:HierarchicalSSM{T}}) where {T} = T
function SSMProblems.arithmetic_type(model::HierarchicalSSM)
    return SSMProblems.arithmetic_type(model.outer_dyn)
end

function AbstractMCMC.sample(
    rng::AbstractRNG, model::HierarchicalSSM, T::Integer; kwargs...
)
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    zs = Vector{eltype(inner_model.dyn)}(undef, T)
    xs = Vector{eltype(outer_dyn)}(undef, T)
    ys = Vector{eltype(inner_model.obs)}(undef, T)

    # Simulate outer dynamics
    x0 = simulate(rng, outer_dyn; kwargs...)
    z0 = simulate(rng, inner_model.dyn; new_outer=x0, kwargs...)
    for t in 1:T
        prev_x = t == 1 ? x0 : xs[t - 1]
        prev_z = t == 1 ? z0 : zs[t - 1]
        xs[t] = simulate(rng, model.outer_dyn, t, prev_x; kwargs...)
        zs[t] = simulate(
            rng, inner_model.dyn, t, prev_z; prev_outer=prev_x, new_outer=xs[t], kwargs...
        )
        ys[t] = simulate(rng, inner_model.obs, t, zs[t]; new_outer=xs[t], kwargs...)
    end

    return x0, z0, xs, zs, ys
end

## Methods to make HierarchicalSSM compatible with the bootstrap filter
struct HierarchicalDynamics{T<:Real,ET,D1<:LatentDynamics{T},D2<:LatentDynamics{T}} <:
       LatentDynamics{T,ET}
    outer_dyn::D1
    inner_dyn::D2
    function HierarchicalDynamics(
        outer_dyn::D1, inner_dyn::D2
    ) where {D1<:LatentDynamics,D2<:LatentDynamics}
        ET = RaoBlackwellisedParticle{eltype(outer_dyn),eltype(inner_dyn)}
        T = SSMProblems.arithmetic_type(outer_dyn)
        return new{T,ET,D1,D2}(outer_dyn, inner_dyn)
    end
end

function SSMProblems.simulate(rng::AbstractRNG, dyn::HierarchicalDynamics; kwargs...)
    outer_dyn, inner_dyn = dyn.outer_dyn, dyn.inner_dyn
    x0 = simulate(rng, outer_dyn; kwargs...)
    z0 = simulate(rng, inner_dyn; new_outer=x0, kwargs...)
    return RaoBlackwellisedParticle(x0, z0)
end

function SSMProblems.simulate(
    rng::AbstractRNG,
    dyn::HierarchicalDynamics,
    step::Integer,
    prev_state::RaoBlackwellisedParticle;
    kwargs...,
)
    outer_dyn, inner_dyn = dyn.outer_dyn, dyn.inner_dyn
    x = simulate(rng, outer_dyn, step, prev_state.x; kwargs...)
    z = simulate(
        rng, inner_dyn, step, prev_state.z; prev_outer=prev_state.x, new_outer=x, kwargs...
    )
    return RaoBlackwellisedParticle(x, z)
end

struct HierarchicalObservations{T<:Real,ET,OP<:ObservationProcess{T}} <:
       ObservationProcess{T,ET}
    obs::OP
    function HierarchicalObservations(obs::OP) where {OP<:ObservationProcess}
        T = SSMProblems.arithmetic_type(obs)
        ET = eltype(obs)
        return new{T,ET,OP}(obs)
    end
end

function SSMProblems.distribution(
    obs::HierarchicalObservations, step::Integer, state::RaoBlackwellisedParticle; kwargs...
)
    return distribution(obs.obs, step, state.z; new_outer=state.x, kwargs...)
end
