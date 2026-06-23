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
# the gh-pages docs. The link targets the `dev` build for a stable URL; note the same
# notebook is also deployed under `previews/PR<n>/` (PR previews) and `vX.Y`/`stable`
# (releases), which this fixed `dev` link does not resolve to.
const REPO = "TuringLang/SSMProblems.jl"
const PKG = basename(dirname(@__DIR__))
const COLAB_BADGE = string(
    "# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]",
    "(https://colab.research.google.com/github/",
    REPO,
    "/blob/gh-pages/",
    PKG,
    "/dev/examples/",
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

# Convert to markdown and notebook
const SCRIPTJL = joinpath(EXAMPLEPATH, "script.jl")
Literate.markdown(SCRIPTJL, OUTDIR; name=EXAMPLE, execute=true, preprocess=add_colab_badge)
# Also emit a runnable notebook; Documenter copies it into the deployed site. The badge is
# only added to the rendered page, not the notebook itself.
Literate.notebook(SCRIPTJL, OUTDIR; name=EXAMPLE, execute=false)
