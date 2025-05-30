push!(LOAD_PATH, "../src/")

#
# With minor changes from https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/docs
#
### Process examples
# Always rerun examples
const EXAMPLES_OUT = joinpath(@__DIR__, "src", "examples")
ispath(EXAMPLES_OUT) && rm(EXAMPLES_OUT; recursive=true)
mkpath(EXAMPLES_OUT)

# Install and precompile all packages
# Workaround for https://github.com/JuliaLang/Pkg.jl/issues/2219
examples = filter!(isdir, readdir(joinpath(@__DIR__, "..", "examples"); join=true))
above = joinpath(@__DIR__, "..")
let script = "using Pkg; Pkg.activate(ARGS[1]); Pkg.develop(path=\"$(above)\"); Pkg.instantiate()"
    for example in examples
        if !success(`$(Base.julia_cmd()) -e $script $example`)
            error(
                "project environment of example ",
                basename(example),
                " could not be instantiated",
            )
        end
    end
end
# Run examples asynchronously
processes = let literatejl = joinpath(@__DIR__, "literate.jl")
    map(examples) do example
        return run(
            pipeline(
                `$(Base.julia_cmd()) $literatejl $(basename(example)) $EXAMPLES_OUT`;
                stdin=devnull,
                stdout=devnull,
                stderr=stderr,
            );
            wait=false,
        )::Base.Process
    end
end

# Check that all examples were run successfully
isempty(processes) || success(processes) || error("some examples were not run successfully")

# Building Documenter
using Documenter
using SSMProblems

DocMeta.setdocmeta!(SSMProblems, :DocTestSetup, :(using SSMProblems); recursive=true)

makedocs(;
    sitename="SSMProblems",
    format=Documenter.HTML(; size_threshold=1000 * 2^11), # 1Mb per page
    #modules=[SSMProblems],
    pages=[
        "Home" => "index.md",
        "Examples" => [
            map(
                (x) -> joinpath("examples", x),
                filter!(filename -> endswith(filename, ".md"), readdir(EXAMPLES_OUT)),
            )...,
        ],
    ],
    #strict=true,
    checkdocs=:exports,
    doctestfilters=[
        # Older versions will show "0 element Array" instead of "Type[]".
        r"(Any\[\]|0-element Array{.+,[0-9]+})",
        # Older versions will show "Array{...,1}" instead of "Vector{...}".
        r"(Array{.+,\s?1}|Vector{.+})",
        # Older versions will show "Array{...,2}" instead of "Matrix{...}".
        r"(Array{.+,\s?2}|Matrix{.+})",
    ],
)
