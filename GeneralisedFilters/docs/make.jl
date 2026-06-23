push!(LOAD_PATH, "../src/")

#
# With minor changes from https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/docs
#
### Process examples
const EXAMPLES_ROOT = joinpath(@__DIR__, "..", "examples")
const EXAMPLES_OUT = joinpath(@__DIR__, "src", "examples")
mkpath(EXAMPLES_OUT)

# Install and precompile all packages
# Workaround for https://github.com/JuliaLang/Pkg.jl/issues/2219
# Only build directories that ship the expected `<slug>.ipynb`; other example dirs
# (e.g. script-only scratch examples) are skipped rather than failing the build.
examples = sort(
    filter(readdir(EXAMPLES_ROOT; join=true)) do path
        return isdir(path) && isfile(joinpath(path, string(basename(path), ".ipynb")))
    end,
)
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
processes = let
    notebookjl = joinpath(@__DIR__, "notebook.jl")
    docs_project = abspath(@__DIR__)
    map(examples) do example
        return run(
            pipeline(
                `$(Base.julia_cmd()) --project=$(docs_project) $notebookjl $(basename(example)) $EXAMPLES_OUT`;
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

const EXAMPLE_MARKDOWNS = sort(
    filter(
        filename -> endswith(filename, ".md") && filename != "index.md",
        readdir(EXAMPLES_OUT),
    ),
)

# Building Documenter
using Documenter
using GeneralisedFilters

DocMeta.setdocmeta!(
    GeneralisedFilters, :DocTestSetup, :(using GeneralisedFilters); recursive=true
)

makedocs(;
    sitename="GeneralisedFilters",
    format=Documenter.HTML(; size_threshold=1000 * 2^11), # 1Mb per page
    pages=[
        "Home" => "index.md",
        "Examples" => map(x -> joinpath("examples", x), EXAMPLE_MARKDOWNS),
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
