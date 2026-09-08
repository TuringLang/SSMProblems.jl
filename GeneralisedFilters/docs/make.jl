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
        "Migrating to 0.6" => "migration.md",
        "API" => "api.md",
    ],
    checkdocs=:exports,
)
