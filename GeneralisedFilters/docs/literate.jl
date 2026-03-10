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
# Pkg.develop(joinpath(@__DIR__, "..", "..", "SSMProblems"))
Pkg.instantiate()
using Literate: Literate

# Determine the package name (GeneralisedFilters or SSMProblems)
const PKG_NAME = if occursin("GeneralisedFilters", abspath(EXAMPLEPATH))
    "GeneralisedFilters"
else
    "SSMProblems"
end

# Git revision for Pkg.add — use the current commit/branch on CI so notebooks match the docs build
const GIT_REV = if haskey(ENV, "GITHUB_ACTIONS")
    ref = get(ENV, "GITHUB_REF", "")
    if startswith(ref, "refs/heads/")
        String(match(r"refs\/heads\/(.*)", ref).captures[1])
    elseif startswith(ref, "refs/tags/")
        String(match(r"refs\/tags\/(.*)", ref).captures[1])
    else
        get(ENV, "GITHUB_SHA", "main")
    end
else
    "main"
end

const REPO_URL = "https://github.com/TuringLang/SSMProblems.jl"

# Compute a version/PR-aware Colab root URL, mirroring Literate.jl's deploy folder logic.
# On CI this produces URLs like:
#   .../blob/gh-pages/GeneralisedFilters/dev/...
#   .../blob/gh-pages/SSMProblems/v1.2.0/...
#   .../blob/gh-pages/GeneralisedFilters/previews/PR42/...
function colab_root_url()
    repo = get(ENV, "GITHUB_REPOSITORY", "TuringLang/SSMProblems.jl")
    deploy_folder = if haskey(ENV, "GITHUB_ACTIONS")
        if get(ENV, "GITHUB_EVENT_NAME", nothing) == "push"
            ref = get(ENV, "GITHUB_REF", "")
            m = match(r"^refs\/tags\/(.*)$", ref)
            m !== nothing ? String(m.captures[1]) : "dev"
        elseif (m = match(r"refs\/pull\/(\d+)\/merge", get(ENV, "GITHUB_REF", ""))) !==
            nothing
            "previews/PR$(m.captures[1])"
        else
            "dev"
        end
    else
        "dev"
    end
    return "https://colab.research.google.com/github/$(repo)/blob/gh-pages/$(PKG_NAME)/$(deploy_folder)"
end

const COLAB_ROOT_URL = colab_root_url()

# Preprocess for markdown: replace @__COLAB_ROOT_URL__ placeholder
function replace_colab_url(content)
    return replace(content, "@__COLAB_ROOT_URL__" => COLAB_ROOT_URL)
end

# Preprocess for notebooks: prepend a Colab-friendly setup cell that installs packages
# and downloads auxiliary files. Uses #nb so these lines only appear in notebooks.
function insert_colab_preamble(content)
    lines = String[]

    push!(lines, "#nb # ## Environment Setup")
    push!(lines, "#nb #")
    push!(
        lines, "#nb # Install required packages (for Google Colab or fresh environments)."
    )
    push!(lines, "#nb import Pkg")
    push!(
        lines,
        "#nb Pkg.add(url=\"$(REPO_URL)\", subdir=\"SSMProblems\", rev=\"$(GIT_REV)\")",
    )

    if PKG_NAME == "GeneralisedFilters"
        push!(
            lines,
            "#nb Pkg.add(url=\"$(REPO_URL)\", subdir=\"GeneralisedFilters\", rev=\"$(GIT_REV)\")",
        )
    end

    # Parse extra dependencies from Project.toml
    project_toml_path = joinpath(EXAMPLEPATH, "Project.toml")
    if isfile(project_toml_path)
        deps = String[]
        in_deps = false
        for line in readlines(project_toml_path)
            if startswith(line, "[deps]")
                in_deps = true
                continue
            elseif startswith(line, "[") && in_deps
                break
            end
            if in_deps && occursin("=", line)
                pkg = strip(split(line, "=")[1])
                if pkg ∉ ("SSMProblems", "GeneralisedFilters", "Literate")
                    push!(deps, "\"$pkg\"")
                end
            end
        end
        if !isempty(deps)
            push!(lines, "#nb Pkg.add([$(join(deps, ", "))])")
        end
    end

    # Download auxiliary files (data, utility scripts) for Colab
    for file in readdir(EXAMPLEPATH)
        if !endswith(file, ".toml") &&
            !isdir(joinpath(EXAMPLEPATH, file)) &&
            !startswith(file, "script")
            url = "https://raw.githubusercontent.com/TuringLang/SSMProblems.jl/$(GIT_REV)/$(PKG_NAME)/examples/$(EXAMPLE)/$(file)"
            push!(lines, "#nb download(\"$(url)\", \"$(file)\")")
            if endswith(file, ".jl")
                push!(lines, "#nb include(\"$(file)\")")
            end
        end
    end

    push!(lines, "")  # blank line before original content
    return join(lines, "\n") * "\n" * content
end

# Postprocess for notebooks: fix kernelspec to be "julia" instead of "julia-1.x" for Colab support
function fix_kernelspec(nb)
    if haskey(nb, "metadata") && haskey(nb["metadata"], "kernelspec")
        nb["metadata"]["kernelspec"]["name"] = "julia"
    end
    return nb
end

# Process the Literate-formatted script in the example directory
let scriptjl = joinpath(EXAMPLEPATH, "script.jl")
    # Generate executed markdown for Documenter (with Colab URL replacement)
    Literate.markdown(
        scriptjl, OUTDIR; name=EXAMPLE, execute=true, preprocess=replace_colab_url
    )
    # Generate notebook with Colab preamble
    Literate.notebook(
        scriptjl, OUTDIR; name=EXAMPLE, execute=true,
        preprocess=insert_colab_preamble, postprocess=fix_kernelspec
    )
end
