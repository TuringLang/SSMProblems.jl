export DiscreteLatentDynamics
export DiscreteStateSpaceModel
export HomogenousDiscretePrior, HomogeneousDiscreteLatentDynamics

import SSMProblems: distribution
import Distributions: Categorical

abstract type DiscretePrior <: StatePrior end
abstract type DiscreteLatentDynamics <: LatentDynamics end

function calc_α0 end
function calc_P end

const DiscreteStateSpaceModel = SSMProblems.StateSpaceModel{
    <:DiscretePrior,<:DiscreteLatentDynamics,<:ObservationProcess
}

#######################
#### DISTRIBUTIONS ####
#######################

function SSMProblems.distribution(prior::DiscretePrior; kwargs...)
    α0 = calc_α0(prior; kwargs...)
    return Categorical(α0)
end

function SSMProblems.distribution(
    dyn::DiscreteLatentDynamics, step::Integer, state::Integer; kwargs...
)
    P = calc_P(dyn, step; kwargs...)
    return Categorical(P[state, :])
end

####################################
#### HOMOGENEOUS DISCRETE MODEL ####
####################################

struct HomogenousDiscretePrior{AT<:AbstractVector} <: DiscretePrior
    α0::AT
end

struct HomogeneousDiscreteLatentDynamics{PT<:AbstractMatrix} <: DiscreteLatentDynamics
    P::PT
end

calc_α0(prior::HomogenousDiscretePrior; kwargs...) = prior.α0
calc_P(dyn::HomogeneousDiscreteLatentDynamics, ::Integer; kwargs...) = dyn.P
