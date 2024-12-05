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

struct MultiTargetDynamics{T<:Real} <: LinearGaussianLatentDynamics{T}
    μ0s::Vector{Vector{T}}
    Σ0s::Vector{Matrix{T}}
    As::Vector{Matrix{T}}
    bs::Vector{Vector{T}}
    Qs::Vector{Matrix{T}}
    N::Int
    D::Int
end

GeneralisedFilters.calc_μ0(dyn::MultiTargetDynamics; kwargs...) = vcat(dyn.μ0s...)
function GeneralisedFilters.calc_Σ0(dyn::MultiTargetDynamics; kwargs...)
    Σ0 = zeros(dyn.D * dyn.N, dyn.D * dyn.N)
    for i in 1:(dyn.N)
        idxs = ((i - 1) * dyn.D + 1):(i * dyn.D)
        Σ0[idxs, idxs] .= dyn.Σ0s[i]
    end
    return Σ0
end
function GeneralisedFilters.calc_A(dyn::MultiTargetDynamics, ::Integer; kwargs...)
    A = zeros(dyn.D * dyn.N, dyn.D * dyn.N)
    for i in 1:(dyn.N)
        idxs = ((i - 1) * dyn.D + 1):(i * dyn.D)
        A[idxs, idxs] .= dyn.As[i]
    end
    return A
end
GeneralisedFilters.calc_b(dyn::MultiTargetDynamics, ::Integer; kwargs...) = vcat(dyn.bs...)
function GeneralisedFilters.calc_Q(dyn::MultiTargetDynamics, ::Integer; kwargs...)
    Q = zeros(dyn.D * dyn.N, dyn.D * dyn.N)
    for i in 1:(dyn.N)
        idxs = ((i - 1) * dyn.D + 1):(i * dyn.D)
        Q[idxs, idxs] .= dyn.Qs[i]
    end
    return Q
end

struct MultiTargetObservations{T<:Real} <: LinearGaussianObservationProcess{T}
    Hs::Vector{Matrix{T}}
    Rs::Vector{Matrix{T}}
    N::Int
    D_x::Int
    D_y::Int
end

function GeneralisedFilters.calc_H(obs::MultiTargetObservations, ::Integer; kwargs...)
    H = zeros(obs.D_y * obs.N, obs.D_x * obs.N)
    for i in 1:(obs.N)
        idxs_x = ((i - 1) * obs.D_x + 1):(i * obs.D_x)
        idxs_y = ((i - 1) * obs.D_y + 1):(i * obs.D_y)
        H[idxs_y, idxs_x] .= obs.Hs[i]
    end
    return H
end
function GeneralisedFilters.calc_c(
    obs::MultiTargetObservations{T}, ::Integer; kwargs...
) where {T}
    return zeros(T, obs.D_y * obs.N)
end
function GeneralisedFilters.calc_R(obs::MultiTargetObservations, ::Integer; kwargs...)
    R = zeros(obs.D_y * obs.N, obs.D_y * obs.N)
    for i in 1:(obs.N)
        idxs = ((i - 1) * obs.D_y + 1):(i * obs.D_y)
        R[idxs, idxs] .= obs.Rs[i]
    end
    return R
end

struct ClutterObservations{T<:Real} <: ObservationProcess{Vector{Vector{T}}}
    λ::T
    v::T
end

function SSMProblems.simulate(
    rng::AbstractRNG, obs::ClutterObservations{T}, ::Integer, ::Nothing; kwargs...
) where {T}
    n = rand(rng, Poisson(obs.λ))
    return [obs.v * (T(2) * rand(rng, T, 2) .- T(1)) for _ in 1:n]
end

# TODO: add detection rate
struct ClutterAndMultiTargetObservations{T} <: ObservationProcess{Vector{Vector{T}}}
    target_obs::MultiTargetObservations{T}
    clutter_obs::ClutterObservations{T}
end

function SSMProblems.simulate(
    rng::AbstractRNG,
    obs::ClutterAndMultiTargetObservations{T},
    t::Integer,
    x::Vector;
    kwargs...,
) where {T}
    target_obs, clutter_obs = obs.target_obs, obs.clutter_obs
    targets = SSMProblems.simulate(rng, target_obs, t, x; kwargs...)
    clutter = SSMProblems.simulate(rng, clutter_obs, t, nothing; kwargs...)
    D, N = target_obs.D_y, target_obs.N
    # TODO: get rid of multi target dynamics and instead call Kalman filter iteratively
    targets = [targets[(D * (i - 1) + 1):(D * i)] for i in 1:N]
    return [targets; clutter]
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
# Initialize an empty array to store the converted measurements
ys = []

# Iterate over each cell in cMeas
for i in 1:T
    # Extract the 2*M vector for the current time step
    measurements = cMeas[i]

    # Reshape the vector to a 2xM matrix and then convert it to a vector of tuples (x, y)
    M = length(measurements) ÷ 2
    reshaped_measurements = reshape(measurements, 2, M)
    time_step_measurements = [Tuple(reshaped_measurements[:, j]) for j in 1:M]

    # Append the measurements for the current time step to ys
    push!(ys, time_step_measurements)
end
# Initialize an empty array to store the converted ground truth
xs = []

# Iterate over each cell in groundTruth
for i in 1:T
    # Extract the 2*M vector for the current time step
    positions = groundTruth[i]

    # Reshape the vector to a 2xM matrix and then convert it to a vector of tuples (x, y)
    M = length(positions) ÷ 2
    reshaped_positions = reshape(positions, 2, M)
    time_step_positions = [Tuple(reshaped_positions[:, j]) for j in 1:M]

    # Append the positions for the current time step to xs
    push!(xs, time_step_positions)
end

SEED = 1234;
# Extract the initial positions for the first time step
#initial_positions = groundTruth[1]

# Reshape the initial positions to a 2xN matrix
#reshaped_initial_positions = reshape(initial_positions, 2, N)

#stateini= [reshaped_initial_positions(:, 1) 0 reshaped_initial_positions(:, 3) 0]
# Initialize μ0s with the extracted positions
#μ0s = [stateini[:, i] for i in 1:N]

# Extract the initial positions for the first time step
initial_positions = groundTruth[1]

# Create the initial state vector for each target with zero velocities at the 2nd and 4th positions
stateini = [
    Float64[initial_positions[1, i], 0.0, initial_positions[2, i], 0.0] for i in 1:N
]

# Initialize μ0s with the state vectors
μ0s = [stateini[i] for i in 1:N]

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
R = Matrix(0.3I, 2, 2);

Σ0s = repeat([Σ0], N)
As = repeat([A], N)
bs = repeat([b], N)
Qs = repeat([Q], N)
Hs = repeat([H], N)
Rs = repeat([R], N)

dyn = MultiTargetDynamics(μ0s, Σ0s, As, bs, Qs, N, 4)
target_obs = MultiTargetObservations(Hs, Rs, N, 4, 2)
clutter_obs = ClutterObservations(λ, v)
obs = ClutterAndMultiTargetObservations(target_obs, clutter_obs)

# HACK: manual specification of types to avoid type promotion code
model = StateSpaceModel{Float64,typeof(dyn),typeof(obs)}(dyn, obs)

# rng = MersenneTwister(SEED)
# x0, xs, ys = sample(rng, model, T)
object_positions = [(getindex(xs[t], i)[1], getindex(xs[t], i)[2]) for t in 1:T, i in 1:N]
object_x_coords = [object_positions[t, i][1] for t in 1:T, i in 1:N]
object_y_coords = [object_positions[t, i][2] for t in 1:T, i in 1:N]

# Print object_x_coords to inspect its value
println("object_x_coords: ", object_x_coords)

# Plot the trajectories of all 7 objects over all 50 time steps
p = plot()
for i in 1:7
    plot!(p, object_x_coords[:, i], object_y_coords[:, i]; label="Object $i Trajectory")
end
xlabel!("X")
ylabel!("Y")
title!("Trajectories of 7 Objects over 50 Time Steps")

# Display the plot
display(p)

#true_x = getindex.(xs, 5)
#true_y = getindex.(xs, 7)
#plot!(true_x, true_y; label="True trajectory", legend=:topleft)

t_test = T
meas_x = getindex.(ys[t_test], 1)
meas_y = getindex.(ys[t_test], 2)
scatter!(meas_x, meas_y; label="Measurements")

ys_converted = [[[pos[1], pos[2]] for pos in time_step] for time_step in ys]
ys = ys_converted

meas_x = getindex.(ys[t_test], 1)
meas_y = getindex.(ys[t_test], 2)
scatter!(meas_x, meas_y; label="Measurements")

###################
#### FILTERING ####
###################
abstract type Associator end

struct GlobalNearestNeighborAssociator <: Associator end

# x_t
# -> x_{t+1}_pred
# -> y_{t+1}_pred = H x_{t+1}_pred (noiseless)
# -> y_{t+1} compare with y_{t+1}_pred

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

# HACK: this should be combined with usual initialisation function which should only use
# dynamics
function initialise(model, filter; kwargs...)
    μ0, Σ0 = GeneralisedFilters.calc_initial(model.dyn; kwargs...)
    return Gaussian(μ0, Σ0)
end

function filter(
    model::StateSpaceModel{<:Any,<:Any,<:ClutterAndMultiTargetObservations},
    assoc_filter::AssociationFilter,
    Ys::Vector{Vector{Vector{Float64}}};
    kwargs...,
)
    state = initialise(model, assoc_filter.filter; kwargs...)
    ll = 0.0
    for (t, ys) in enumerate(Ys)
        state, l = step(model, assoc_filter, t, state, ys; kwargs...)
        ll += l
    end
    return state, ll
end

history = Vector{Vector{Float64}}(undef, T)

function step(
    model::StateSpaceModel{<:Any,<:Any,<:ClutterAndMultiTargetObservations},
    assoc_filter::AssociationFilter,
    t::Integer,
    state,
    ys::Vector{Vector{Float64}};
    kwargs...,
)
    # HACK: need a clean way to pass conditioned model to the analytical filter
    # HACK: manual specification of types to avoid type promotion code
    conditioned_model = StateSpaceModel{
        Float64,typeof(model.dyn),typeof(model.obs.target_obs)
    }(
        model.dyn, model.obs.target_obs
    )

    state = GeneralisedFilters.predict(
        conditioned_model, assoc_filter.filter, t, state; kwargs...
    )
    # HACK: assuming form of state
    N, D = model.obs.target_obs.N, model.obs.target_obs.D_x
    μ_vec_vec = [state.μ[(D * (i - 1) + 1):(D * i)] for i in 1:N]
    y_preds = model.obs.target_obs.Hs .* μ_vec_vec
    # y_pred = simulate(model.obs, t, state.μ, extra, noise=false)  # H * x or H * x + R
    assocs = associate(assoc_filter.associator, y_preds, ys)
    state, ll = GeneralisedFilters.update(
        conditioned_model, assoc_filter.filter, t, state, vcat(ys[assocs]...); kwargs...
    )

    history[t] = copy(state.μ)

    return state, ll
end

# Test filter runs
assoc_filter = AssociationFilter(GlobalNearestNeighborAssociator(), KalmanFilter())
state, ll = filter(model, assoc_filter, ys)

# Plot the trajectories of all N=7 objects

# Plot the trajectories of all N=7 objects
p = plot()
# Plot the true trajectories with solid lines in black
for i in 1:N
    plot!(
        p,
        object_x_coords[:, i],
        object_y_coords[:, i];
        label=(i == 1 ? "Ground Truth" : ""),
        color=:black,
    )
end
# Plot the estimated trajectories with dashed lines
for i in 1:N
    filt_x = getindex.(history, 4 * (i - 1) + 1)
    filt_y = getindex.(history, 4 * (i - 1) + 3)
    plot!(p, filt_x, filt_y; label=(i == 1 ? "Estimate" : ""), linestyle=:dash, linewidth=3)
end

meas_x = getindex.(ys[t_test], 1)
meas_y = getindex.(ys[t_test], 2)
scatter!(
    meas_x,
    meas_y;
    label="Measurements",
    color=:grey,
    marker=:circle,
    markersize=2,
    markerstrokecolor=:grey,
)

xlabel!("X")
ylabel!("Y")
title!("Trajectories of $N Objects over $T Time Steps")

# Save the plot as a figure
savefig(p, joinpath(base_path, "trajectories_plot.png"))

# Display the plot
display(p)
