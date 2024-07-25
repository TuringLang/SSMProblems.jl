"""
    @default_arg f(a = 1, b) = ...

Generate an additional method for a function with the first argument replaced by a default
value.

Used in SSMProblems.jl to allow all sampling methods to be called without a RNG, in which
case the `default_rng()` is used.

# Examples
```julia-repl
julia> using Random
julia> @default_arg f(rng::AbstractRNG=Random.default_rng(), x) = x * rand(rng);
julia> methods(f)
# 2 methods for generic function "f" from Main:
 [1] f(b)
     @ none:0
 [2] f(rng, b)
     @ REPL[2]:1
"""
macro default_arg(ex)
    (ex.head === :function || ex.head === :(=)) ||
        throw(ArgumentError("Invalid expression"))
    call = ex.args[1]
    call1 = copy(call)
    call2 = copy(call1)
    Nargs = length(call.args)
    for i in 2:Nargs
        arg = call.args[i]
        if Meta.isexpr(arg, :kw, 2)
            call.args[i] = arg.args[1]
            deleteat!(call1.args, i)
            call2.args[i] = arg.args[2]
            break
        end
    end
    q = quote
        $ex
        $call1 = $call2
    end
    return esc(q)
end
