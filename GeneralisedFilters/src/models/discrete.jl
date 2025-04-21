export DiscreteLatentDynamics
export DiscreteStateSpaceModel
export HomogeneousDiscreteLatentDynamics

import SSMProblems: distribution
import Distributions: Categorical

abstract type DiscreteLatentDynamics{T_state<:Integer,T_prob<:Real} <:
              LatentDynamics{T_prob,T_state} end

function calc_α0 end
function calc_P end

const DiscreteStateSpaceModel{T} = SSMProblems.StateSpaceModel{
    T,LD,OD
} where {T,LD<:DiscreteLatentDynamics{<:Integer,T},OD<:ObservationProcess{T}}

function rb_eltype(
    ::DiscreteStateSpaceModel{LD}
) where {T_state,T_prob,LD<:DiscreteLatentDynamics{T_state,T_prob}}
    return Vector{T_prob}
end

#######################
#### DISTRIBUTIONS ####
#######################

function SSMProblems.distribution(dyn::DiscreteLatentDynamics; kwargs...)
    α0 = calc_α0(dyn; kwargs...)
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

# TODO: likewise, where do these types come from?
struct HomogeneousDiscreteLatentDynamics{T_state<:Integer,T_prob<:Real} <:
       DiscreteLatentDynamics{T_state,T_prob}
    α0::Vector{T_prob}
    P::Matrix{T_prob}
end
calc_α0(dyn::HomogeneousDiscreteLatentDynamics; kwargs...) = dyn.α0
calc_P(dyn::HomogeneousDiscreteLatentDynamics, ::Integer; kwargs...) = dyn.P
