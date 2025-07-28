import SSMProblems: LatentDynamics, ObservationProcess, simulate
export HierarchicalSSM

struct HierarchicalSSM{PT<:StatePrior,LD<:LatentDynamics,MT<:StateSpaceModel} <:
       AbstractStateSpaceModel
    outer_prior::PT
    outer_dyn::LD
    inner_model::MT
end

function HierarchicalSSM(
    outer_prior::StatePrior,
    outer_dyn::LatentDynamics,
    inner_prior::StatePrior,
    inner_dyn::LatentDynamics,
    obs::ObservationProcess,
)
    inner_model = StateSpaceModel(inner_prior, inner_dyn, obs)
    return HierarchicalSSM(outer_prior, outer_dyn, inner_model)
end

function AbstractMCMC.sample(
    rng::AbstractRNG, model::HierarchicalSSM, T::Integer; kwargs...
)
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    xs = OffsetVector(fill(simulate(rng, model.outer_prior; kwargs...), T + 1), -1)
    zs = OffsetVector(
        fill(simulate(rng, inner_model.prior; new_outer=xs[0], kwargs...), T + 1), -1
    )

    # Simulate outer dynamics
    xs[0] = simulate(rng, outer_dyn; kwargs...)
    zs[0] = simulate(rng, inner_model.dyn; new_outer=xs[0], kwargs...)
    for t in 1:T
        xs[t] = simulate(rng, model.outer_dyn, t, xs[t - 1]; kwargs...)
        zs[t] = simulate(
            rng,
            inner_model.dyn,
            t,
            zs[t - 1];
            prev_outer=xs[t - 1],
            new_outer=xs[t],
            kwargs...,
        )
    end

    ys = map(t -> simulate(rng, inner_model.obs, t, zs[t]; new_outer=xs[t], kwargs...), 1:T)
    return xs, zs, ys
end

## Methods to make HierarchicalSSM compatible with the bootstrap filter
struct HierarchicalDynamics{D1<:LatentDynamics,D2<:LatentDynamics} <: LatentDynamics
    outer_dyn::D1
    inner_dyn::D2
end

struct HierarchicalPrior{P1<:StatePrior,P2<:StatePrior} <: StatePrior
    outer_prior::P1
    inner_prior::P2
end

function SSMProblems.simulate(rng::AbstractRNG, prior::HierarchicalPrior; kwargs...)
    outer_prior, inner_prior = prior.outer_prior, prior.inner_prior
    x0 = simulate(rng, outer_prior; kwargs...)
    z0 = simulate(rng, inner_prior; new_outer=x0, kwargs...)
    return RaoBlackwellisedParticle(x0, z0)
end

function SSMProblems.simulate(
    rng::AbstractRNG,
    proc::HierarchicalDynamics,
    step::Integer,
    prev_state::RaoBlackwellisedParticle;
    kwargs...,
)
    outer_dyn, inner_dyn = proc.outer_dyn, proc.inner_dyn
    x = simulate(rng, outer_dyn, step, prev_state.x; kwargs...)
    z = simulate(
        rng, inner_dyn, step, prev_state.z; prev_outer=prev_state.x, new_outer=x, kwargs...
    )
    return RaoBlackwellisedParticle(x, z)
end

struct HierarchicalObservations{OP<:ObservationProcess} <: ObservationProcess
    obs::OP
end

function SSMProblems.distribution(
    obs::HierarchicalObservations, step::Integer, state::RaoBlackwellisedParticle; kwargs...
)
    return distribution(obs.obs, step, state.z; new_outer=state.x, kwargs...)
end
