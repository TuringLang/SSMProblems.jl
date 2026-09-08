# Run in an environment containing GeneralisedFilters, ForwardDiff and Mooncake:
# julia --project=<environment> GeneralisedFilters/benchmark/rbpg.jl
# Reports warm evaluation cost separately from gradient preparation. This is a
# baseline measurement, not an allocation assertion or a full Turing Gibbs benchmark.
using GeneralisedFilters
using StaticArrays
using Random
using Statistics
using DifferentiationInterface
using ADTypes
using ForwardDiff
using Mooncake

function measure(label, f; repetitions=10)
    f() # compile and warm reusable buffers
    GC.gc()
    measurements = [@timed(f()) for _ in 1:repetitions]
    println(
        label,
        ": median ",
        round(1e6 * median(m.time for m in measurements); digits=2),
        " μs; ",
        median(m.bytes for m in measurements),
        " bytes/evaluation",
    )
    return nothing
end

function run_benchmarks(; horizon=100, particles=100)
    rng = MersenneTwister(123)
    # Stable dynamics keep a long trajectory in a representative numeric range.
    outer_prior = GaussianPrior(SA[0.0], SMatrix{1,1}(0.5))
    outer_dyn = LinearGaussianDynamics(SMatrix{1,1}(0.9), SA[0.0], SMatrix{1,1}(0.05))
    inner_prior = GaussianPrior(SA[0.0, 0.0], SA[1.0 0.0; 0.0 1.0])
    function build(θ)
        q = exp(θ[1]) * SA[0.1 0.01; 0.01 0.2]
        r = exp(θ[2]) * SMatrix{1,1}(0.3)
        dynamics =
            ctx -> LinearGaussianDynamics(SA[0.8 0.1; 0.0 0.9], SA[ctx.x_prev[1], 0.0], q)
        observation = LinearGaussianObservation(SMatrix{1,2}(1.0, 0.5), SA[0.0], r)
        return StateSpaceModel(outer_prior, outer_dyn, inner_prior, dynamics, observation)
    end
    θ = [0.0, 0.0]
    model = build(θ)
    s0, ss, ys = simulate(rng, model, horizon)
    xs = ReferenceTrajectory(s0.x, getproperty.(ss, :x))
    algo = RBPF(BF(particles), KF())
    particle = GeneralisedFilters.initialise_particle(rng, model.prior, algo, nothing)
    println(
        "Julia ",
        VERSION,
        "; horizon=",
        horizon,
        ", particles=",
        particles,
        ", inner dimension=2, parameters=2",
    )
    println(
        "ForwardDiff ",
        pkgversion(ForwardDiff),
        "; Mooncake ",
        pkgversion(Mooncake),
        "; DifferentiationInterface ",
        pkgversion(DifferentiationInterface),
        "; StaticArrays ",
        pkgversion(StaticArrays),
    )
    println("Particle state isbits: ", isbitstype(typeof(particle.state)))
    measure(
        "One static RB particle prediction",
        () -> GeneralisedFilters.predict_particle(
            rng, model.dyn, algo, 1, particle, ys[1], nothing
        ),
    )
    measure("RBPF full trajectory", () -> GeneralisedFilters.filter(rng, model, algo, ys))
    objective = θ -> trajectory_logdensity(build(θ), KF(), xs, ys)
    for backend in (AutoForwardDiff(), AutoMooncake(; config=nothing))
        preparation = @timed prepare_gradient(objective, backend, θ)
        println(
            typeof(backend),
            " preparation: ",
            round(preparation.time; digits=3),
            " s; ",
            preparation.bytes,
            " bytes",
        )
        measure(
            "$(typeof(backend)) prepared gradient",
            () -> gradient(objective, preparation.value, backend, θ),
        )
    end
end

run_benchmarks()
