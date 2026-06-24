# Retrieve name of example and output directory
if length(ARGS) != 2
    error("please specify the name of the example and the output directory")
end
const EXAMPLE = ARGS[1]
const OUTDIR = ARGS[2]

# Activate environment
# Note that each example's Project.toml must include Literate as a dependency
using Pkg: Pkg
const EXAMPLEPATH = joinpath(@__DIR__, "..", "examples", EXAMPLE)
Pkg.activate(EXAMPLEPATH)
Pkg.instantiate()
using Literate: Literate

# Insert an "Open in Colab" badge (after the title) that opens the notebook published to
# the gh-pages docs. The version subdir matches Documenter's deploy target: `previews/PR<n>`
# for pull-request previews, the tag (e.g. `v1.2.3`) for tagged releases, otherwise `dev`.
const REPO = "TuringLang/SSMProblems.jl"
const PKG = basename(dirname(@__DIR__))

function docs_subfolder()
    ref = get(ENV, "GITHUB_REF", "")
    if get(ENV, "GITHUB_EVENT_NAME", "") == "pull_request"
        pr = match(r"^refs/pull/(\d+)/", ref)
        pr === nothing || return string("previews/PR", only(pr.captures))
    end
    tag = match(r"^refs/tags/(.+)$", ref)
    tag === nothing || return only(tag.captures)
    return "dev"
end

const COLAB_BADGE = string(
    "# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]",
    "(https://colab.research.google.com/github/",
    REPO,
    "/blob/gh-pages/",
    PKG,
    "/",
    docs_subfolder(),
    "/examples/",
    EXAMPLE,
    ".ipynb)",
)

function add_colab_badge(content)
    lines = string.(split(content, '\n'))
    idx = findfirst(line -> startswith(line, "# # "), lines)
    idx === nothing && return string(COLAB_BADGE, "\n#\n", content)
    insert!(lines, idx + 1, "#")
    insert!(lines, idx + 2, COLAB_BADGE)
    return join(lines, '\n')
end

const SCRIPTJL = joinpath(EXAMPLEPATH, "script.jl")

# The example's declared dependencies (Literate is build-only), so the notebook can install
# everything it needs on Colab — including packages used only by included helper files.
function example_packages()
    deps = get(Pkg.TOML.parsefile(joinpath(EXAMPLEPATH, "Project.toml")), "deps", Dict())
    return sort(filter(!=("Literate"), collect(keys(deps))))
end

# Notebook postprocess: use a generic Julia kernel (so Colab picks its default Julia
# runtime, with no pinned version) and prepend a single Pkg.add for the example's
# dependencies so the notebook runs on a fresh Colab runtime.
function prepare_notebook(nb)
    nb["metadata"]["kernelspec"] = Dict(
        "display_name" => "Julia", "language" => "julia", "name" => "julia"
    )
    # `#hide` is a Documenter-only directive that Literate leaves in the notebook; drop
    # those lines (e.g. the repo-relative paths that only work during the docs build).
    for cell in nb["cells"]
        cell["cell_type"] == "code" || continue
        cell["source"] = filter(line -> !endswith(rstrip(line), "#hide"), cell["source"])
    end
    packages = example_packages()
    if !isempty(packages)
        setup = Dict(
            "cell_type" => "code",
            "execution_count" => nothing,
            "metadata" => Dict(),
            "outputs" => [],
            "source" =>
                ["import Pkg\n", string("Pkg.add([", join(repr.(packages), ", "), "])")],
        )
        idx = findfirst(cell -> cell["cell_type"] == "code", nb["cells"])
        insert!(nb["cells"], something(idx, lastindex(nb["cells"]) + 1), setup)
    end
    return nb
end

# Convert to markdown and notebook
Literate.markdown(SCRIPTJL, OUTDIR; name=EXAMPLE, execute=true, preprocess=add_colab_badge)
# Also emit a runnable notebook; Documenter copies it into the deployed site, under the
# versioned docs dir (dev/, vX.Y/, or previews/PR<n>/ for PR previews). The badge goes on
# the rendered page only; the notebook gets a Colab setup cell instead.
Literate.notebook(
    SCRIPTJL, OUTDIR; name=EXAMPLE, execute=false, postprocess=prepare_notebook
)
