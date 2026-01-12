module Models

using AbstractMCMC
using SSMProblems

export ParameterisedSSM

"""
    ParameterisedSSM(model_builder, θ_prior)

A wrapper for a state space model where some parameters θ are unknown and follow a prior.
`model_builder` is a function θ -> StateSpaceModel.
"""
struct ParameterisedSSM{M, P} <: AbstractMCMC.AbstractModel
    model_builder::M
    prior::P
end

end
