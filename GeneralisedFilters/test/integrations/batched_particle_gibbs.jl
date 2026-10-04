@testitem "GPU particle Gibbs uses CPU conditional targets and current parameters" tags = [
    :gpu, :batched
] begin
    include(joinpath(@__DIR__, "..", "..", "examples", "gpu-rbpf", "model.jl"))
    using .GPUVolatilityExample, CUDA, BatchedKernels, Random, Distributions, ForwardDiff
    using AbstractMCMC: AbstractMCMC
    using AdvancedHMC: NUTS
    const GF = GeneralisedFilters
    CUDA.allowscalar(false)
    fixed = GPUVolatilityExample.fixed_parameters(2, 1, Float32)
    build(θ) = GPUVolatilityExample.model(fixed, only(θ))
    observations = GPUVolatilityExample.observations(build(Float32[0.1]), 3)
    param_model = ParameterisedSSM(build, [CuArray(Vector(y)) for y in observations])
    reference = ReferenceTrajectory(
        CuArray(Float32[0.2]), [CuArray(Float32[x]) for x in (-0.3, 0.7, 0.1)]
    )
    host_model, host_path = GF._parameter_inputs(GPUExecution(), param_model, reference)
    @test host_model.build === build
    @test all(y -> y isa Vector{Float32}, host_model.observations)
    @test all(x -> x isa Vector{Float32}, host_path)
    @test firstindex(host_path) == 0
    # Parameter changes must affect the target without rebuilding fixed device arrays.
    target(a) =
        GF.trajectory_logdensity(build([a]), KF(), host_path, host_model.observations)
    a, h = 0.1, 1e-5
    @test ForwardDiff.derivative(target, a) ≈ (target(a+h)-target(a-h))/(2h) rtol=1e-5

    model = ParticleGibbsModel(MvNormal(Float32[0], Float32[0.25;;]), param_model)
    pf = RBPF(BF(17; execution=GPUExecution()), KF())
    sampler = ParticleGibbs(ConditionalSMC(pf, BackwardSimulation()), NUTS(0.8f0))
    rng = CombinedRNG(Xoshiro(51), BatchedRNG(52))
    draw, state = AbstractMCMC.step(
        rng, model, sampler; initial_params=Float32[0.1], n_adapts=5
    )
    second, state = AbstractMCMC.step(rng, model, sampler, state; n_adapts=5)
    @test all(isfinite, second.θ)
    @test all(x -> x isa CUDA.AnyCuVector{Float32}, state.trajectory)
    @test haskey(second.stat, :acceptance_rate)
end
