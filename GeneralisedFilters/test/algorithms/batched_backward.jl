@testitem "Batched RBPF backward likelihoods, refreshment and smoothing" tags = [
    :gpu, :batched
] begin
    include(joinpath(@__DIR__, "..", "..", "examples", "gpu-rbpf", "model.jl"))
    using .GPUVolatilityExample, CUDA, BatchedKernels, Random
    const GF=GeneralisedFilters
    CUDA.allowscalar(false)
    begin
        model=GPUVolatilityExample.model(2, 1)
        cy=GPUVolatilityExample.observations(model, 3)
        ys=[CuArray(Vector(y)) for y in cy]
        cr=ReferenceTrajectory(Float32[0.2], [Float32[x] for x in (-0.3, 0.7, 0.1)])
        gr=map(CuArray, cr)
        pf=RBPF(BF(17; threshold=1.0, execution=GPUExecution()), KF())
        gl=GF._compute_backward_likelihoods(
            CUDA.RNG(5), model, pf, ys, gr, AncestorSampling()
        )
        cl=GF._compute_backward_likelihoods(
            Xoshiro(5), model, pf, cy, cr, AncestorSampling()
        )
        # CPU suffix messages are uploaded only when scoring the device population.
        @test all(l -> !(l.B isa CUDA.AnyCuArray) && !(l.r isa CUDA.AnyCuArray), gl)
        for (g, c) in zip(gl, cl)
            @test Array(g.B)'*Array(g.B) ≈ c.B'*c.B rtol=2.0f-4 atol=2.0f-5
            @test Array(g.B)'*Array(g.r) ≈ c.B'*c.r rtol=2.0f-4 atol=2.0f-5
            @test g.logscale-sum(abs2, Array(g.r))/2 ≈ c.logscale-sum(abs2, c.r)/2 rtol=2.0f-4 atol=2.0f-5
        end
        state=initialise(CUDA.RNG(9), model.prior, pf)
        state, _=GF.step(CUDA.RNG(10), model, pf, 1, state, ys[1])
        parents = state.particles.components.state.components.x
        dynamics = GF.inner_dynamics(model.dyn, 2, parents, gr[2])
        @test dynamics.components.Q isa SharedCuMatrix
        @test Array(dynamics.components.Q.data) ≈
            exp(only(cr[2])) * model.dyn.inner.parameters.host.Q
        ws=GF._ancestor_weights(state, model.dyn, pf, 2, RBState(gr[2], gl[2]))
        fields=state.particles.components
        x=Array(fields.state.components.x.data)
        μ=Array(fields.state.components.z.components.μ.data)
        Σ=Array(fields.state.components.z.components.Σ.data)
        w=Array(fields.log_w.data)
        expected=[
            GF.ancestor_weight(
                GF.Particle(RBState(x[:, i], GaussianState(μ[:, i], Σ[:, :, i])), w[i], i),
                model.dyn,
                pf,
                2,
                RBState(cr[2], cl[2]),
            ) for i in 1:17
        ]
        @test GF.softmax(Array(ws)) ≈ GF.softmax(expected) rtol=3.0f-4 atol=2.0f-5
        # Exercise the public conditional model and shared RTS time loop.
        gs, gll = GF.smooth(CUDA.RNG(12), condition_inner(model, gr), KS, ys)
        cs, cll = GF.smooth(Xoshiro(12), condition_inner(model, cr), KS, cy)
        @test Array(gs.μ) ≈ cs.μ rtol=2.0f-4 atol=2.0f-5
        @test Array(gs.Σ) ≈ cs.Σ rtol=2.0f-4 atol=2.0f-5
        @test gll ≈ cll rtol=2.0f-4 atol=2.0f-5
    end

    @testset "GPU conditional refreshment" begin
        for strategy in (AncestorSampling(), BackwardSimulation())
            algo=ConditionalSMC(
                RBPF(BF(17; threshold=1.0, execution=GPUExecution()), KF()), strategy
            )
            model=GPUVolatilityExample.model(2, 1)
            ys=[CuArray(Vector(y)) for y in GPUVolatilityExample.observations(model, 3)]
            traj, ll=GF._csmc_sample(CUDA.RNG(44), model, algo, ys, nothing)
            @test isfinite(ll)
            traj, ll=GF._csmc_sample(CUDA.RNG(45), model, algo, ys, traj)
            @test isfinite(ll)
            @test length(traj)==4
        end
    end
end

@testitem "One model selects CPU or GPU populations and differentiates on CPU" tags = [
    :gpu, :batched
] begin
    include(joinpath(@__DIR__, "..", "..", "examples", "gpu-rbpf", "model.jl"))
    using .GPUVolatilityExample, CUDA, BatchedKernels, Random, ForwardDiff
    const GF = GeneralisedFilters
    CUDA.allowscalar(false)
    fixed = GPUVolatilityExample.fixed_parameters(2, 1, Float64)
    model = GPUVolatilityExample.model(fixed, 0.1)
    reference = ReferenceTrajectory([0.2], [[-0.3], [0.7], [0.1]])
    ys = [[0.1], [-0.2], [0.3]]
    rng = CombinedRNG(Xoshiro(11), BatchedRNG(12))
    # One conditioned particle removes Monte Carlo differences between RNG backends.
    cpu, cpu_ll = GF.filter(rng, model, RBPF(BF(1), KF()), ys; ref_state=reference)
    gpu, gpu_ll = GF.filter(
        rng,
        model,
        RBPF(BF(1; execution=GPUExecution()), KF()),
        map(CuArray, ys);
        ref_state=map(CuArray, reference),
    )
    z = gpu.particles.components.state.components.z
    @test Array(z.components.μ.data)[:, 1] ≈ cpu.particles[1].state.z.μ
    @test Array(z.components.Σ.data)[:, :, 1] ≈ cpu.particles[1].state.z.Σ
    @test gpu_ll ≈ cpu_ll
    objective(a) =
        GF.trajectory_logdensity(GPUVolatilityExample.model(fixed, a), KF(), reference, ys)
    h = 1e-5
    @test ForwardDiff.derivative(objective, 0.1) ≈
        (objective(0.1 + h) - objective(0.1 - h)) / (2h) rtol=1e-6
end
