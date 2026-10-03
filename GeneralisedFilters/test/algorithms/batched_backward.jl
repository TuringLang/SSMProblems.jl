@testitem "Batched RBPF backward likelihoods, refreshment and smoothing" tags = [
    :gpu, :batched
] begin
    include(joinpath(@__DIR__, "..", "..", "examples", "gpu-rbpf", "model.jl"))
    using .GPUVolatilityExample, CUDA, BatchedKernels, Random
    const GF=GeneralisedFilters
    CUDA.allowscalar(false)
    begin
        cpu, gpu=GPUVolatilityExample.models(2, 1)
        cy=GPUVolatilityExample.observations(cpu, 3)
        ys=[CuArray(Vector(y)) for y in cy]
        cr=ReferenceTrajectory(Float32[0.2], [Float32[x] for x in (-0.3, 0.7, 0.1)])
        gr=map(CuArray, cr)
        pf=RBPF(BF(17; threshold=1.0), KF())
        gl=GF._compute_backward_likelihoods(
            CUDA.RNG(5), gpu, pf, ys, gr, AncestorSampling()
        )
        cl=GF._compute_backward_likelihoods(Xoshiro(5), cpu, pf, cy, cr, AncestorSampling())
        for (g, c) in zip(gl, cl)
            @test Array(g.B)'*Array(g.B) ≈ c.B'*c.B rtol=2.0f-4 atol=2.0f-5
            @test Array(g.B)'*Array(g.r) ≈ c.B'*c.r rtol=2.0f-4 atol=2.0f-5
            @test g.logscale-sum(abs2, Array(g.r))/2 ≈ c.logscale-sum(abs2, c.r)/2 rtol=2.0f-4 atol=2.0f-5
        end
        state=initialise(CUDA.RNG(9), gpu.prior, pf)
        state, _=GF.step(CUDA.RNG(10), gpu, pf, 1, state, ys[1])
        ws=GF._ancestor_weights(state, gpu.dyn, pf, 2, RBState(gr[2], gl[2]))
        fields=state.particles.components
        x=Array(fields.state.components.x.data)
        μ=Array(fields.state.components.z.components.μ.data)
        Σ=Array(fields.state.components.z.components.Σ.data)
        w=Array(fields.log_w.data)
        expected=[
            GF.ancestor_weight(
                GF.Particle(RBState(x[:, i], GaussianState(μ[:, i], Σ[:, :, i])), w[i], i),
                cpu.dyn,
                pf,
                2,
                RBState(cr[2], cl[2]),
            ) for i in 1:17
        ]
        @test GF.softmax(Array(ws)) ≈ GF.softmax(expected) rtol=3.0f-4 atol=2.0f-5
        # Exercise the public conditional model and shared RTS time loop.
        gs, gll = GF.smooth(CUDA.RNG(12), condition_inner(gpu, gr), KS, ys)
        cs, cll = GF.smooth(Xoshiro(12), condition_inner(cpu, cr), KS, cy)
        @test Array(gs.μ) ≈ cs.μ rtol=2.0f-4 atol=2.0f-5
        @test Array(gs.Σ) ≈ cs.Σ rtol=2.0f-4 atol=2.0f-5
        @test gll ≈ cll rtol=2.0f-4 atol=2.0f-5
    end

    @testset "GPU conditional refreshment" begin
        for strategy in (AncestorSampling(), BackwardSimulation())
            algo=ConditionalSMC(RBPF(BF(17; threshold=1.0), KF()), strategy)
            cpu, gpu=GPUVolatilityExample.models(2, 1)
            ys=[CuArray(Vector(y)) for y in GPUVolatilityExample.observations(cpu, 3)]
            traj, ll=GF._csmc_sample(CUDA.RNG(44), gpu, algo, ys, nothing)
            @test isfinite(ll)
            traj, ll=GF._csmc_sample(CUDA.RNG(45), gpu, algo, ys, traj)
            @test isfinite(ll)
            @test length(traj)==4
        end
    end
end
