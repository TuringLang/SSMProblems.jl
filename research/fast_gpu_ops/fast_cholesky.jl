using BenchmarkTools
using CUDA
using LinearAlgebra
using Plots
using ProgressMeter

BATCH_SIZES = 10 .^ (1:7)
D = 2

####################
#### GENERATORS ####
####################

function random_psd(d)
    A = rand(d, d)
    return A * A'
end

function random_psd_batch(d, bs)
    return [random_psd(d) for _ in 1:bs]
end

function random_psd_strided_batch(d, bs)
    As = Array{Float64}(undef, d, d, bs)
    for i in 1:bs
        As[:, :, i] = random_psd(d)
    end
    return As
end

function random_psd_cuda(d)
    A = random_psd(d)
    return cu(A)
end

function random_psd_cuda_batch(d, bs)
    return [random_psd_cuda(d) for _ in 1:bs]
end

function random_psd_cuda_strided_batch(d, bs)
    As = CUDA.rand(d, d, bs)
    return As = CUDA.CUBLAS.gemm_strided_batched('N', 'T', 1.0f0, As, As)
end

#################
#### METHODS ####
#################

function cpu_native(As::Vector{Matrix{Float64}})
    return [cholesky(A) for A in As]
end

function cpu_2d_analytical(As::Array{Float64,3})
    out = Array{Float64}(undef, size(As))
    out[1, 1, :] .= sqrt.(As[1, 1, :])
    out[2, 1, :] .= As[2, 1, :] ./ out[1, 1, :]
    out[1, 2, :] .= out[2, 1, :]
    out[2, 2, :] .= sqrt.(As[2, 2, :] .- out[2, 1, :] .^ 2)
    return out
end

function gpu_2d_analytical(As::CuArray{Float32,3})
    out = CUDA.zeros(Float32, size(As))
    out[1, 1, :] .= sqrt.(As[1, 1, :])
    out[2, 1, :] .= As[2, 1, :] ./ out[1, 1, :]
    out[1, 2, :] .= out[2, 1, :]
    out[2, 2, :] .= sqrt.(As[2, 2, :] .- out[2, 1, :] .^ 2)
    return out
end

##################
#### VALIDATE ####
##################

###################
#### BENCHMARK ####
###################

function benchmark(d, batch_sizes, methods)
    results = Dict{String,Vector{BenchmarkTools.Trial}}()
    @showprogress for (method, initialiser) in methods
        method_results = Vector{BenchmarkTools.Trial}(undef, length(batch_sizes))
        for (i, bs) in enumerate(batch_sizes)
            As = initialiser(d, bs)
            method_results[i] = @benchmark CUDA.@sync ($method)($As)
        end
        results[String(Symbol(method))] = method_results
    end
    return results
end

benchmark_methods = Dict{Function,Function}(cpu_native => random_psd_batch)

# 2D analytical methods
if D == 2
    push!(benchmark_methods, cpu_2d_analytical => random_psd_strided_batch)
    push!(benchmark_methods, gpu_2d_analytical => random_psd_cuda_strided_batch)
end

# Run benchmark
results = benchmark(D, BATCH_SIZES, benchmark_methods)

###################
#### VISUALISE ####
###################

p = plot(; yscale=:log10, xaxis=:log10, xlabel="Batch size", ylabel="Time (s)")
for (method, trials) in results
    plot!(p, BATCH_SIZES, median.(getfield.(trials, :times)) / 10^9; label=string(method))
end
plot(p)
