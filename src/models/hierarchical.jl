import SSMProblems: LatentDynamics, ObservationProcess, simulate
export HierarchicalSSM

struct HierarchicalSSM{LD1<:LatentDynamics,LD2<:LatentDynamics,OP<:ObservationProcess} <:
       SSMProblems.AbstractStateSpaceModel
    outer_dyn::LD1
    inner_dyn::LD2
    obs::OP
end

function AbstractMCMC.sample(
    rng::AbstractRNG, model::HierarchicalSSM, extras::AbstractVector
)
    T = length(extras)
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    # Simulate outer dynamics
    xs = Vector{typeof(x0)}(undef, T)
    x0 = simulate(rng, outer_dyn)
    augmented_extras = Vector{NamedTuple{}}(undef, T)
    for t in 1:T
        prev_x = t == 1 ? x0 : xs[t - 1]
        xs[t] = simulate(rng, model.outer_dyn, t, prev_x, extras[t])
        new_extras = (prev_outer=prev_x, new_outer=xs[t])
        augmented_extras[t] = isnothing(extras[t]) ? new_extras : (extras[t]..., new_extras)
    end

    # Simulate inner model
    zs, ys = sample(rng, inner_model, augmented_extras)

    return xs, zs, ys
end
