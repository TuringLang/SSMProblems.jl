# Linear Gaussian Models

GeneralisedFilters is designed to work with the most general form of linear
Gaussian state models. This can be written as:

$$
\begin{align*}
x_0 &\sim \mathcal{N}(\mu_0, \Sigma_0) \\
x_t &= A_t x_{t-1} + b_t + w_t & w_t &\sim \mathcal{N}(0, Q_t) \\
y_t &= H_t x_t + c_t + v_t & v_t &\sim \mathcal{N}(0, R_t)
\end{align*}
$$

where the model parameters $A_t, b_t, Q_t, H_t, c_t, R_t$ can be constant,
time-varying, or even functions of exogenous inputs. For example, we could have
$b_t = B u_t$ for some input $u_t$ and matrix $B$.

Most of the time, you will not need this generality, but it is still there in
case you do!

In GeneralisedFilters.jl, a linear Gaussian state space model is just a type
that can return these model parameters given a time index (and possibly an input).

## Homogeneous (Non-Time Varying) Linear Gaussian Models

If you wish to define a simple linear Gaussian model with constant parameters,
you can use the concrete types:

- `HomogeneousGaussianPrior`
- `HomogeneousLinearGaussianDynamics`
- `HomogeneousLinearGaussianObservations` 

These accept `AbstractMatrix` and `AbstractVector` values for the model
parameters, which will be applied at each timestep. For example, we might have,

```julia
μ0 = [1.0 2.0]
Σ0 = [1.0 0.0; 0.0 1.0]
my_prior = HomogeneousGaussianPrior(μ0, Σ0)
```

Importantly, the `HomogeneousXYZ` types are parameterised on the types of each
of the fields. For example `HomogeneousLinearGaussianLatentDynamics` is defined as:

```julia
struct HomogeneousLinearGaussianLatentDynamics{
    AT<:AbstractMatrix,bT<:AbstractVector,QT<:AbstractMatrix
} <: LinearGaussianLatentDynamics
    A::AT
    b::bT
    Q::QT
end
```

This allows you to use specialised types for the model parameters that may
capture additional structure to allow for more efficient computations. You can
also take advantage of [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl)
for lightning-fast filtering in low-dimensional settings. As an illustrative
example you could have:

```
using StaticArrays
using SparseArrays
using PDMats

A = @SMatrix [1.0 0.1; 0.0 0.7]
b = sparse([1.0, 0.0])
Q = PDiagMat([0.1, 0.5])

my_dynamics = HomogeneousLinearGaussianLatentDynamics(A, b, Q)
```

GeneralisedFilters.jl's linear Gaussian filtering methods have been written in
such a way to preserve StaticArrays.

If your application is particularly performance-sensitive and your model does
not make use of $b_t$ and/or $c_t$, you can replace these with lazy
representation of a zero vector from [FillArrays.jl](https://github.com/JuliaArrays/FillArrays.jl):

```
using FillArrays
HomogeneousLinearGaussianObservations(H, Zeros(dim_y), R)
```

## General Linear Gaussian Models

GeneralisedFilters.jl defines linear Gaussian models in a fairly abstract fashion.
Although this might seem a bit overkill at first glance, it leads to some
powerful benefits in terms of modularity and extensibility. 

We define the components of a linear Gaussian state space model as subtypes of
the following three abstract types:

- `GaussianPrior`  (itself a subtype of `StatePrior`)
- `LinearGaussianLatentDynamics` (a subtype of `LatentDynamics`)
- `LinearGaussianObservationProcess` (a subtype of `ObservationProcess`)

These must define `calc_` functions that return the model parameters for a
given time index. For example, a general `LinearGaussianLatentDynamics` must
define

```julia
calc_A(dyn::MyDynamicsType, step::Integer; kwargs...)
```

The `calc_` functions for the prior are `calc_μ0` and `calc_Σ0` and these do not
take a time index argument.

The `kwargs...` argument is included to allow for exogenous inputs to be
passed to the model. For example, in the case where `b_t = B u_t`, we could define

```julia
function calc_b(dyn::MyDynamicsType, step::Integer; u)
    return dyn.B * u[step]
end
```

where `u` is a vector of inputs passed to the filtering/smoothing function.

Finally, in cases where it is more efficient/convenient to compute the various
model parameters at the same time (e.g. `A_t` and `b_t` depend on a common value
that is expensive to compute), you can write a method for the following
functions which return a tuple of the relevant parameters:

- `calc_initial(prior::GaussianPrior; kwargs...)` returns `(μ0, Σ0)`
- `calc_dynamics(dyn::LinearGaussianLatentDynamics, step::Integer; kwargs...)`
  returns `(A, b, Q)`
- `calc_observations(obs::LinearGaussianObservationProcess, step::Integer; kwargs...)`
  returns `(H, c, R)`
