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

# Notebook preprocessor: prepend a Colab-friendly setup cell that installs packages
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
        "#nb Pkg.add(url=\"https://github.com/TuringLang/SSMProblems.jl\", subdir=\"SSMProblems\")",
    )

    if occursin("GeneralisedFilters", abspath(EXAMPLEPATH))
        push!(
            lines,
            "#nb Pkg.add(url=\"https://github.com/TuringLang/SSMProblems.jl\", subdir=\"GeneralisedFilters\")",
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
    pkg_subdir =
        occursin("GeneralisedFilters", EXAMPLEPATH) ? "GeneralisedFilters" : "SSMProblems"
    for file in readdir(EXAMPLEPATH)
        if !endswith(file, ".toml") &&
            !isdir(joinpath(EXAMPLEPATH, file)) &&
            !startswith(file, "script")
            url = "https://raw.githubusercontent.com/TuringLang/SSMProblems.jl/main/$(pkg_subdir)/examples/$(EXAMPLE)/$(file)"
            push!(lines, "#nb download(\"$(url)\", \"$(file)\")")
            if endswith(file, ".jl")
                push!(lines, "#nb include(\"$(file)\")")
            end
        end
    end

    push!(lines, "")  # blank line before original content
    return join(lines, "\n") * "\n" * content
end

# Process all Literate-formatted scripts in the example directory
for filename in readdir(EXAMPLEPATH)
    if endswith(filename, ".jl") &&
        filename != "utilities.jl" &&
        startswith(filename, "script")
        scriptjl = joinpath(EXAMPLEPATH, filename)
        # Name mapping: script.jl -> example_name, script-alt.jl -> example_name-alt
        name = replace(splitext(filename)[1], "script" => EXAMPLE)
        # Generate executed markdown for Documenter
        Literate.markdown(scriptjl, OUTDIR; name=name, execute=true)
        # Generate notebook (not executed — user runs it interactively or on Colab)
        Literate.notebook(
            scriptjl, OUTDIR; name=name, execute=false, preprocess=insert_colab_preamble
        )
    end
end
