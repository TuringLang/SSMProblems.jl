# Build one example from notebook source into markdown for Documenter.
if length(ARGS) != 2
    error("please specify the name of the example and the output directory")
end

const EXAMPLE = ARGS[1]
const OUTDIR = ARGS[2]
const REPO = "TuringLang/SSMProblems.jl"
const PKG_SUBDIR = "GeneralisedFilters"
const DOCS_ENV = abspath(@__DIR__)

using Pkg: Pkg

const EXAMPLEPATH = joinpath(@__DIR__, "..", "examples", EXAMPLE)
const NOTEBOOK_FILENAME = string(EXAMPLE, ".ipynb")
const NOTEBOOK = joinpath(EXAMPLEPATH, NOTEBOOK_FILENAME)
const MARKDOWN = joinpath(OUTDIR, string(EXAMPLE, ".md"))

isfile(NOTEBOOK) || error("example $(EXAMPLE) must include $(NOTEBOOK_FILENAME)")

# Keep notebook execution in the example's own environment.
Pkg.activate(EXAMPLEPATH)
Pkg.instantiate()

Pkg.activate(DOCS_ENV)
Pkg.instantiate()
using IJulia

const KERNEL_NAME = "gf-docs-julia-$(VERSION.major).$(VERSION.minor)"
IJulia.installkernel(KERNEL_NAME, "--project=$(DOCS_ENV)")

function run_nbconvert(
    examplepath::AbstractString,
    outdir::AbstractString,
    name::AbstractString,
    kernel::AbstractString,
)
    jupyter = Sys.which("jupyter")
    isnothing(jupyter) && error(
        "jupyter executable not found. Install it (e.g. `pip install jupyter nbconvert`) before building docs.",
    )

    cmd = `$(jupyter) nbconvert --to markdown --execute --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=$(kernel) --output=$(name) --output-dir=$(outdir) $(NOTEBOOK_FILENAME)`
    run(pipeline(Cmd(cmd; dir=examplepath); stdin=devnull, stdout=devnull, stderr=stderr))
    return nothing
end

function inject_edit_url(markdown_path::AbstractString, example::AbstractString)
    content = read(markdown_path, String)
    meta_block = string(
        "```@meta\n",
        "EditURL = \"../../../examples/",
        example,
        "/",
        NOTEBOOK_FILENAME,
        "\"\n",
        "```\n\n",
    )

    if occursin(r"(?m)^```@meta$", content)
        content = replace(content, r"(?s)^```@meta.*?```\s*" => meta_block; count=1)
    else
        content = string(meta_block, content)
    end

    write(markdown_path, content)
    return nothing
end

function inject_docs_badges(markdown_path::AbstractString, example::AbstractString)
    content = read(markdown_path, String)
    notebook_name = string(example, ".ipynb")

    colab_url = string(
        "https://colab.research.google.com/github/",
        REPO,
        "/blob/main/",
        PKG_SUBDIR,
        "/examples/",
        example,
        "/",
        notebook_name,
    )
    source_url = string(
        "https://github.com/",
        REPO,
        "/blob/main/",
        PKG_SUBDIR,
        "/examples/",
        example,
        "/",
        notebook_name,
    )
    badge_line = string(
        "[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](",
        colab_url,
        ") [![View Source](https://img.shields.io/badge/View%20Source-GitHub-181717?logo=github)](",
        source_url,
        ")",
    )

    # Remove existing notebook badge lines, then inject docs-specific badges.
    content = replace(content, r"(?m)^\[\!\[Open in Colab\].*\n?" => "";)
    content = replace(content, r"\n{3,}" => "\n\n";)

    heading_match = match(r"(?m)^# .+$", content)
    if isnothing(heading_match)
        content = string(badge_line, "\n\n", content)
    else
        head_start = heading_match.offset
        head_end = head_start + ncodeunits(heading_match.match) - 1
        head = content[1:head_end]
        tail = content[(head_end + 1):end]
        tail = replace(tail, r"^\n+" => "")
        content = string(head, "\n\n", badge_line, "\n\n", tail)
    end

    write(markdown_path, content)
    return nothing
end

run_nbconvert(EXAMPLEPATH, OUTDIR, EXAMPLE, KERNEL_NAME)
inject_docs_badges(MARKDOWN, EXAMPLE)
inject_edit_url(MARKDOWN, EXAMPLE)
