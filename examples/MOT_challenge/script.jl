using AbstractMCMC
using GaussianDistributions
using GeneralisedFilters
using Distributions
using LinearAlgebra
using Plots
using Random
using SSMProblems
using Hungarian
using MAT

import GeneralisedFilters: AbstractFilter

base_path = "examples/MOT_challenge/"

abstract type AbstractClutter end

struct TargetsAndClutter{M<:AbstractStateSpaceModel,C<:AbstractClutter}
    target_models::Vector{M}
    clutter_model::C
end

function sample(
    rng::AbstractRNG, clutter_model::TargetsAndClutter{M}, T::Integer; kwargs...
) where {LD,OP,M<:StateSpaceModel{<:Any,LD,OP}}
    N = length(clutter_model.target_models)
    T_dyn = eltype(LD)
    T_obs = eltype(OP)

    xs = Vector{Vector{T_dyn}}(undef, T)
    ys = Vector{Vector{T_obs}}(undef, T)

    # Initialisation
    x0 = [sample(rng, model.dyn, 1; kwargs...) for model in clutter_model.target_models]

    for t in 1:T
        step_xs = Vector{T_dyn}(undef, N)
        step_ys = Vector{T_obs}(undef, N)
        for (i, model) in enumerate(clutter_model.target_models)
            step_xs[i] = simulate(rng, model.dyn, t, t == 1 ? x0 : xs[t - 1]; kwargs...)
            step_ys[i] = simulate(rng, model.obs, t, step_xs[i]; kwargs...)
        end
        # Add clutter
        clutter = simulate(rng, clutter_model.clutter_model, t; kwargs...)
        append!(step_ys, clutter)

        xs[t] = step_xs
        ys[t] = step_ys
    end

    return x0, xs, ys
end

struct UniformClutter{T<:Real} <: AbstractClutter
    λ::T
    v::T
end

function SSMProblems.simulate(
    rng::AbstractRNG, cltr::UniformClutter{T}, ::Integer; kwargs...
) where {T}
    n = rand(rng, Poisson(obs.λ))
    return [obs.v * (T(2) * rand(rng, T, 2) .- T(1)) for _ in 1:n]
end

# Load the .mat file
matfile = matread(joinpath(base_path, "mot15PETS09_S2L1_turing.mat"))

# Access the 'cMeas' variable
cMeas = matfile["cMeas"]
groundTruth = matfile["x"]
T = Int(matfile["T"])

N = 7
v = matfile["area"]
# pd = 0.9
λ = matfile["aveNumClt"]

# Load measurements
ys = Vector{Vector{Vector{Float64}}}(undef, T)
for i in 1:T
    # Extract the 2M vector for the current time step
    measurements = cMeas[i]

    # Convert flat array into vector of vectors
    time_step_measurements = [measurements[j:(j + 1)] for j in 1:2:length(measurements)]

    # Add the measurements for the current time step to ys
    ys[i] = time_step_measurements
end

# Load ground truth
xs = Vector{Vector{Vector{Float64}}}(undef, T)
for i in 1:T
    # Extract the 2M vector for the current time step
    positions = groundTruth[i]

    # Convert flat array into vector of vectors
    time_step_positions = [positions[j:(j + 1)] for j in 1:2:length(positions)]

    # Add the positions for the current time step to xs
    xs[i] = time_step_positions
end

SEED = 1234;

# Initialize μ0s at the ground truth positions with zero velocity
μ0s = [[xs[1][i][1], 0.0, xs[1][i][2], 0.0] for i in 1:N]
Σ0 = Matrix(1.0I, 4, 4);
A = [1.0 1.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 1.0; 0.0 0.0 0.0 1.0];
b = zeros(4);
Q = [
    1/3 1/2 0 0
    1/2 1 0 0
    0 0 1/3 1/2
    0 0 1/2 1
];
H = [1.0 0.0 0.0 0.0; 0.0 0.0 1.0 0.0];
c = zeros(2);
R = Matrix(0.3I, 2, 2);

ssms = [create_homogeneous_linear_gaussian_model(μ0s[i], Σ0, A, b, Q, H, c, R) for i in 1:N]

clutter_obs = UniformClutter(λ, v)

model = TargetsAndClutter(ssms, clutter_obs)

# Plot the trajectories of all 7 objects over all 50 time steps
p = plot(;
    xlabel="X",
    ylabel="Y",
    title="Trajectories of 7 Objects over 50 Time Steps",
    aspect_ratio=:equal,
    legend=:outerbottom,
    legend_columns=3,
    size=(800, 800),
)
for i in 1:7
    traj = getindex.(xs, i)
    plot!(p, getindex.(traj, 1), getindex.(traj, 2); label="Object $i", linewidth=2)
end
t_test = T
meas_x = getindex.(ys[t_test], 1)
meas_y = getindex.(ys[t_test], 2)
scatter!(p, meas_x, meas_y; label="Measurements")
display(p)

###################
#### FILTERING ####
###################

abstract type Associator end

struct GlobalNearestNeighborAssociator <: Associator end

function associate(::GlobalNearestNeighborAssociator, y_preds::Vector, ys::Vector)
    n_preds = length(y_preds)
    n_ys = length(ys)

    # Create the cost matrix where cost(i, j) = euclidean_distance(y_preds[i], ys[j])
    cost_matrix = zeros(Float64, n_preds, n_ys)
    for i in 1:n_preds
        for j in 1:n_ys
            cost_matrix[i, j] = norm(y_preds[i] - ys[j])
        end
    end

    # Solve the assignment problem using the Hungarian algorithm
    row_indices, col_indices = hungarian(cost_matrix)

    # Prepare the associations
    assocs = row_indices
    return assocs
end

struct AssociationFilter{A<:Associator,F<:AbstractFilter} <: AbstractFilter
    associator::A
    filter::F
end

function GeneralisedFilters.initialise(
    clutter_model::TargetsAndClutter{<:Any,<:Any},
    assoc_filter::AssociationFilter;
    kwargs...,
)
    states = [
        GeneralisedFilters.initialise(model, assoc_filter.filter; kwargs...) for
        model in clutter_model.target_models
    ]
    return states
end

function filter(
    clutter_model::TargetsAndClutter{<:Any,<:Any},
    assoc_filter::AssociationFilter,
    Ys::Vector{Vector{Vector{Float64}}};
    kwargs...,
)
    states = GeneralisedFilters.initialise(clutter_model, assoc_filter; kwargs...)
    lls = zeros(Float64, length(clutter_model.target_models))
    for (t, ys) in enumerate(Ys)
        states, ls = step(clutter_model, assoc_filter, t, states, ys; kwargs...)
        lls += ls
    end
    return states, lls
end

function step(
    clutter_model::TargetsAndClutter,
    assoc_filter::AssociationFilter,
    t::Integer,
    states,
    ys::Vector{Vector{Float64}};
    # HACK: need to combine this with callbacks
    history,
    kwargs...,
)
    algo = assoc_filter.filter
    # Predict step
    for (i, model) in enumerate(clutter_model.target_models)
        states[i] = GeneralisedFilters.predict(model, algo, t, states[i]; kwargs...)
    end
    # Association step
    # HACK: manually computing predicted observation to avoid adding noise
    y_preds = [
        model.obs.H * state.μ for (model, state) in zip(clutter_model.target_models, states)
    ]
    assocs = associate(assoc_filter.associator, y_preds, ys)
    # update step
    lls = Vector{Float64}(undef, length(states))
    for (i, model) in enumerate(clutter_model.target_models)
        states[i], lls[i] = GeneralisedFilters.update(
            model, algo, t, states[i], ys[assocs[i]]; kwargs...
        )
    end

    history[t] = deepcopy(states)
    return states, lls
end

# Test filter runs
assoc_filter = AssociationFilter(GlobalNearestNeighborAssociator(), KalmanFilter())
history = Vector{Vector{Gaussian{Vector{Float64},Matrix{Float64}}}}(undef, T)
state, ll = filter(model, assoc_filter, ys; history=history)

# Add estimated trajectories to the plot as dotted lines
for i in 1:N
    traj = getproperty.(getindex.(history, i), :μ)
    plot!(
        p,
        getindex.(traj, 1),
        getindex.(traj, 3);  # second entry is velocity
        label="Object $i (Filtered)",
        linewidth=2,
        linestyle=:dash,
        color=:black,
    )
end
display(p)
