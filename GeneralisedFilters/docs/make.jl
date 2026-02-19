push!(LOAD_PATH, "../src/")
const REPO = "TuringLang/SSMProblems.jl"
const PKG_SUBDIR = "GeneralisedFilters"

#
# With minor changes from https://github.com/JuliaGaussianProcesses/AbstractGPs.jl/docs
#
### Process examples
# Always rerun examples
const EXAMPLES_ROOT = joinpath(@__DIR__, "..", "examples")
const EXAMPLES_OUT = joinpath(@__DIR__, "src", "examples")
ispath(EXAMPLES_OUT) && rm(EXAMPLES_OUT; recursive=true)
mkpath(EXAMPLES_OUT)
const EXAMPLE_ASSETS_OUT = joinpath(@__DIR__, "src", "assets", "examples")
mkpath(EXAMPLE_ASSETS_OUT)

# Install and precompile all packages
# Workaround for https://github.com/JuliaLang/Pkg.jl/issues/2219
examples = sort(filter!(isdir, readdir(EXAMPLES_ROOT; join=true)))
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
processes = let notebookjl = joinpath(@__DIR__, "notebook.jl")
    map(examples) do example
        return run(
            pipeline(
                `$(Base.julia_cmd()) $notebookjl $(basename(example)) $EXAMPLES_OUT`;
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

example_slug(markdown_filename::AbstractString) = splitext(markdown_filename)[1]

function example_title(markdown_path::AbstractString, slug::AbstractString)
    for line in eachline(markdown_path)
        stripped = strip(line)
        startswith(stripped, "# ") && return strip(stripped[3:end])
    end
    return replace(slug, "-" => " ")
end

function example_summary(markdown_path::AbstractString)
    in_fence = false
    for line in eachline(markdown_path)
        stripped = strip(line)

        if startswith(stripped, "```")
            in_fence = !in_fence
            continue
        end

        if in_fence || isempty(stripped)
            continue
        end

        if startswith(stripped, "# ")
            continue
        end

        if occursin("Open in Colab", stripped) ||
           occursin("Source notebook", stripped) ||
           startswith(stripped, "*This page was generated")
            continue
        end

        return replace(stripped, "|" => "\\|")
    end

    return "Runnable example with executable source."
end

function example_thumbnail(slug::AbstractString)
    for ext in ("svg", "png", "jpg", "jpeg", "webp")
        thumb = joinpath(EXAMPLE_ASSETS_OUT, string(slug, ".", ext))
        if isfile(thumb)
            return "../assets/examples/$(slug).$(ext)"
        end
    end
    return "../assets/examples/default.svg"
end

function links_for_example(slug::AbstractString)
    example_dir = joinpath(EXAMPLES_ROOT, slug)
    notebook_name = string(slug, ".ipynb")
    isfile(joinpath(example_dir, notebook_name)) ||
        error("example $(slug) must include $(notebook_name)")

    return join(
        [
            "[Colab](https://colab.research.google.com/github/$(REPO)/blob/main/$(PKG_SUBDIR)/examples/$(slug)/$(notebook_name))",
            "[Notebook](https://github.com/$(REPO)/blob/main/$(PKG_SUBDIR)/examples/$(slug)/$(notebook_name))",
        ],
        " · ",
    )
end

function write_examples_index(example_markdowns::Vector{String})
    index_path = joinpath(EXAMPLES_OUT, "index.md")
    open(index_path, "w") do io
        println(io, "# Examples")
        println(io)
        println(
            io,
            "Executable examples for `GeneralisedFilters` with links to notebooks and source files.",
        )
        println(io)
        println(io, "| Example | Preview |")
        println(io, "| :-- | :-- |")
        for markdown in example_markdowns
            slug = example_slug(markdown)
            markdown_path = joinpath(EXAMPLES_OUT, markdown)
            title = example_title(markdown_path, slug)
            summary = example_summary(markdown_path)
            links = links_for_example(slug)
            thumbnail = example_thumbnail(slug)

            page_link = "[$(title)]($(markdown))"
            left = string(page_link, "<br>", summary, "<br>", links)
            right = "[![$(title)]($(thumbnail))]($(markdown))"
            println(io, "| $(left) | $(right) |")
        end
    end
    return nothing
end

const EXAMPLE_MARKDOWNS = sort(
    filter(
        filename ->
            endswith(filename, ".md") && filename != "index.md", readdir(EXAMPLES_OUT),
    ),
)
write_examples_index(EXAMPLE_MARKDOWNS)

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
        "Examples" => Any[
            "examples/index.md",
            map((x) -> joinpath("examples", x), EXAMPLE_MARKDOWNS)...,
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
