import SSMProblems:
    LatentDynamics, ObservationProcess, AbstractStateSpaceModel, StateSpaceModel, simulate

export HierarchicalSSM

struct HierarchicalSSM{OD<:LatentDynamics,M<:AbstractStateSpaceModel} <:
       AbstractStateSpaceModel
    outer_dyn::OD
    inner_model::M
end
function HierarchicalSSM(
    outer_dyn::LatentDynamics, inner_dyn::LatentDynamics, obs::ObservationProcess
)
    return HierarchicalSSM(outer_dyn, StateSpaceModel(inner_dyn, obs))
end

function AbstractMCMC.sample(
    rng::AbstractRNG, model::HierarchicalSSM, extras::AbstractVector
)
    T = length(extras)

    x0_outer = simulate(rng, model.outer_dyn, extras[1])
    augmented_extra = (; extras[1]..., outer=x0_outer)
    x0_inner = simulate(rng, model.inner_dyn, augmented_extra)
    y0 = simulate(rng, model.obs, 1, x0_inner, augmented_extra)

    x0 = (; outer=x0_outer, inner=x0_inner)

    xs = Vector{typeof(x0)}(undef, T)
    ys = Vector{typeof(y0)}(undef, T)

    xs[1] = x0
    ys[1] = y0

    for t in 2:T
        x_outer = simulate(rng, model.outer_dyn, t, xs[t - 1].outer, extras[t])
        augmented_extra = (; extras[t]..., outer=x_outer)
        x_inner = simulate(rng, model.inner_dyn, t, xs[t - 1].inner, augmented_extra)
        y = simulate(rng, model.obs, t, x_inner, augmented_extra)

        xs[t] = (; outer=x_outer, inner=x_inner)
        ys[t] = y
    end

    return xs, ys
end

# TODO: refactor this
# export InnerDynamics
