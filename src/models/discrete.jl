export DiscreteDynamics
export DiscreteStateSpaceModel
export HomogeneousDiscreteLatentDynamics

import SSMProblems: distribution
import Distributions: Categorical

abstract type DiscreteLatentDynamics{T_state<:Integer,T_prob<:Real} <:
              SSMProblems.LatentDynamics{T_state} end

function calc_α0 end
function calc_P end

const DiscreteStateSpaceModel{LD,OD} = SSMProblems.StateSpaceModel{
    T,LD,OD
} where {T,LD<:DiscreteLatentDynamics,OD<:ObservationProcess{T}}

function rb_eltype(
    ::DiscreteStateSpaceModel{LD}
) where {T_state,T_prob,LD<:DiscreteLatentDynamics{T_state,T_prob}}
    return Vector{T_prob}
end

#######################
#### DISTRIBUTIONS ####
#######################

function SSMProblems.distribution(dyn::DiscreteLatentDynamics, extra=nothing; kwargs...)
    α0 = calc_α0(dyn; kwargs...)
    return Categorical(α0)
end

function SSMProblems.distribution(
    dyn::DiscreteLatentDynamics{T}, step::Integer, state::Integer, extra=nothing; kwargs...
) where {T}
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
