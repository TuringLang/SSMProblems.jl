import SSMProblems: LatentDynamics, ObservationProcess, simulate
export HierarchicalSSM

struct HierarchicalSSM{T<:Real,OD<:LatentDynamics,M<:AbstractStateSpaceModel} <:
       AbstractStateSpaceModel
    outer_dyn::OD
    inner_model::M
end

function HierarchicalSSM(
    outer_dyn::LatentDynamics{LDT}, inner_dyn::LatentDynamics, obs::ObservationProcess
) where {LDT}
    inner_model = StateSpaceModel(inner_dyn, obs)
    T = promote_type(eltype(inner_model), eltype(LDT))
    return HierarchicalSSM{T,typeof(outer_dyn),typeof(inner_model)}(outer_dyn, inner_model)
end

Base.eltype(::Type{<:HierarchicalSSM{T,ODT,MT}}) where {T,ODT,MT} = T

function AbstractMCMC.sample(
    rng::AbstractRNG, model::HierarchicalSSM, extra0, extras::AbstractVector
)
    T = length(extras)
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    xs = Vector{eltype(outer_dyn)}(undef, T)
    augmented_extras = Vector{NamedTuple}(undef, T)

    # Simulate outer dynamics
    x0 = simulate(rng, outer_dyn, extra0)
    for t in 1:T
        prev_x = t == 1 ? x0 : xs[t - 1]
        xs[t] = simulate(rng, model.outer_dyn, t, prev_x, extras[t])
        new_extras = (prev_outer=prev_x, new_outer=xs[t])
        augmented_extras[t] =
            isnothing(extras[t]) ? new_extras : (; extras[t]..., new_extras...)
    end

    new_extra0 = (; new_outer=x0)
    augmented_extra0 = isnothing(extra0) ? new_extra0 : (; extra0..., new_extra0...)

    # Simulate inner model
    z0, zs, ys = sample(rng, inner_model, augmented_extra0, augmented_extras)

    return x0, z0, xs, zs, ys
end
