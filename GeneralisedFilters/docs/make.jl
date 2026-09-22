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
# Only build directories that ship a Literate `script.jl`; other example dirs
# (e.g. script-only scratch examples) are skipped rather than failing the build.
examples = filter(readdir(joinpath(@__DIR__, "..", "examples"); join=true)) do path
    return isdir(path) && isfile(joinpath(path, "script.jl"))
end
above = joinpath(@__DIR__, "..")
ssmproblems_path = joinpath(above, "..", "SSMProblems")
let script = "using Pkg; Pkg.activate(ARGS[1]); Pkg.develop(path=\"$(above)\"); Pkg.develop(path=\"$(ssmproblems_path)\"); Pkg.instantiate()"
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

using Documenter
using GeneralisedFilters

DocMeta.setdocmeta!(
    GeneralisedFilters, :DocTestSetup, :(using GeneralisedFilters); recursive=true
)
makedocs(;
    sitename="GeneralisedFilters",
    modules=[GeneralisedFilters],
    format=Documenter.HTML(),
    pages=[
        "Overview" => "index.md",
        "Models and conditioning" => "models/linear-gaussian.md",
        "Particle Gibbs and Turing" => "inference.md",
        "Migrating to 0.5" => "migration.md",
        "Examples" =>
            [joinpath("examples", f) for f in readdir(EXAMPLES_OUT) if endswith(f, ".md")],
        "API" => "api.md",
    ],
    checkdocs=:exports,
)
