# RBPG building-block benchmark

Run `rbpg.jl` from an environment with GeneralisedFilters, ForwardDiff and Mooncake.
The script reports preparation separately from repeated prepared gradients, and checks
whether static particle prediction allocates. It measures the conditional objective and
ordinary RBPF; it does not measure complete Turing particle-Gibbs sweeps.

An indicative Julia 1.12.7 run on an AMD Ryzen Threadripper 7960X used ForwardDiff
1.4.5, Mooncake 0.5.53, DifferentiationInterface 0.7.21 and StaticArrays 1.9.20.
With 100 observations, 100 particles, two inner dimensions, one outer dimension and two
parameters, it gave:

| Operation | Median time | Allocated bytes |
|:--|--:|--:|
| One static particle prediction | 0.15 μs | 0 |
| Full RBPF | 2.65 ms | 3,583,784 |
| Prepared ForwardDiff gradient | 15.48 μs | 80 |
| Prepared Mooncake gradient | 24.04 μs | 160 |

First gradient preparation, including compilation, took 0.107 seconds/9.12 MB for
ForwardDiff and 28.77 seconds/2.51 GB for Mooncake. These are machine- and
session-dependent measurements, affected by compilation and concurrent hardware load,
not performance guarantees or test thresholds. They do not establish the
parameter dimension at which reverse mode wins. The tests additionally compare both AD
backends on a 20-parameter conditional objective.

The full filter still allocates particle and resampling storage. The integration currently
rebuilds AD preparation after each trajectory change to ensure a correct target. Workspace
reuse and large-parameter crossover measurements remain useful follow-up work.
