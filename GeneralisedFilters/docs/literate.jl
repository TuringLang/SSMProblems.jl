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

function insert_colab_preamble(content)
    # Extract dependencies from Project.toml
    project_toml_path = joinpath(EXAMPLEPATH, "Project.toml")
    preamble = """
    #nb # This cell specifies dependencies for the notebook to run in environments like Google Colab.
    #nb import Pkg
    #nb Pkg.add(url="https://github.com/TuringLang/SSMProblems.jl", subdir="SSMProblems")
    #nb """
    
    if occursin("GeneralisedFilters", abspath(EXAMPLEPATH))
        preamble *= "Pkg.add(url=\"https://github.com/TuringLang/SSMProblems.jl\", subdir=\"GeneralisedFilters\")\n    #nb "
    end

    if isfile(project_toml_path)
        deps_lines = String[]
        in_deps = false
        for line in readlines(project_toml_path)
            if startswith(line, "[deps]")
                in_deps = true
                continue
            elseif startswith(line, "[") && in_deps
                in_deps = false
                continue
            end
            if in_deps && occursin("=", line)
                pkg = strip(split(line, "=")[1])
                if pkg != "SSMProblems" && pkg != "GeneralisedFilters" && pkg != "Literate"
                    push!(deps_lines, "\"$pkg\"")
                end
            end
        end
        if !isempty(deps_lines)
            preamble *= "Pkg.add([$(join(deps_lines, ", "))])\n"
        end
    end
    
    # Also download extra files if they exist (e.g., data.csv, utilities.jl)
    for file in readdir(EXAMPLEPATH)
        if file != "script.jl" && file != "script-alt.jl" && !endswith(file, ".toml") && !isdir(joinpath(EXAMPLEPATH, file))
            # Determine base URL dynamically based on whether we're in GeneralisedFilters or SSMProblems
            pkg_subdir = occursin("GeneralisedFilters", EXAMPLEPATH) ? "GeneralisedFilters" : "SSMProblems"
            url = "https://raw.githubusercontent.com/TuringLang/SSMProblems.jl/main/\$(pkg_subdir)/examples/\$(EXAMPLE)/\$file"
            preamble *= "    #nb download(\"$url\", \"\$file\")\n"
            if endswith(file, ".jl")
                preamble *= "    #nb include(\"\$file\")\n"
            end
        end
    end
    
    return preamble * "\n" * content
end

# Convert to markdown and notebook
for filename in readdir(EXAMPLEPATH)
    if endswith(filename, ".jl") && filename != "utilities.jl"
        SCRIPTJL = joinpath(EXAMPLEPATH, filename)
        # Map script -> example_name, script-alt -> example_name-alt
        name = replace(splitext(filename)[1], "script" => EXAMPLE)
        Literate.markdown(SCRIPTJL, OUTDIR; name=name, execute=true)
        Literate.notebook(SCRIPTJL, OUTDIR; name=name, execute=true, preprocess=insert_colab_preamble)
    end
end
