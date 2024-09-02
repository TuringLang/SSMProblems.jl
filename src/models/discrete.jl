export DiscreteDynamics
export DiscreteStateSpaceModel
export HomogeneousDiscreteLatentDynamics

import SSMProblems: distribution
import Distributions: Categorical

abstract type DiscreteLatentDynamics{T<:Integer} <: SSMProblems.LatentDynamics{T} end

function calc_α0 end
function calc_P end

const DiscreteStateSpaceModel{T} = SSMProblems.StateSpaceModel{
    D,O
} where {T,D<:DiscreteLatentDynamics{T},O<:ObservationProcess{T}}

# TODO: how do we inference this type? Depends on the type of α0/P
rb_eltype(::DiscreteStateSpaceModel) = Vector{Float64}

#######################
#### DISTRIBUTIONS ####
#######################

function SSMProblems.distribution(dyn::DiscreteLatentDynamics)
    α0 = calc_α0(dyn)
    return Categorical(α0)
end

function SSMProblems.distribution(
    dyn::DiscreteLatentDynamics{T}, step::Integer, state::Integer, extra
) where {T}
    P = calc_P(dyn, step, extra)
    return Categorical(P[state, :])
end

####################################
#### HOMOGENEOUS DISCRETE MODEL ####
####################################

# TODO: likewise, where do these types come from?
struct HomogeneousDiscreteLatentDynamics{T<:Integer} <: DiscreteLatentDynamics{T}
    α0::Vector{Float64}
    P::Matrix{Float64}
end
calc_α0(dyn::HomogeneousDiscreteLatentDynamics) = dyn.α0
calc_P(dyn::HomogeneousDiscreteLatentDynamics, ::Integer, extra) = dyn.P
