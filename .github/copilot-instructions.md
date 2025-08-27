# SSMProblems.jl Development Guide

**ALWAYS follow these instructions first and fallback to additional search and context gathering only if the information in these instructions is incomplete or found to be in error.**

## Repository Overview

SSMProblems.jl is a Julia package repository containing two packages:
- **SSMProblems**: Core package defining interfaces for state space models 
- **GeneralisedFilters**: Filtering algorithms package that depends on SSMProblems

## Working Effectively

### Essential Setup Commands
Run these commands to bootstrap your development environment:

```bash
# Install Julia dependencies for SSMProblems
cd SSMProblems
julia --project -e "using Pkg; Pkg.instantiate()"

# Install Julia dependencies for GeneralisedFilters (includes CUDA stack)
cd ../GeneralisedFilters  
julia --project -e "using Pkg; Pkg.develop(path=\"../SSMProblems\"); Pkg.instantiate()"
```

**NEVER CANCEL**: GeneralisedFilters dependency installation takes 3+ minutes due to CUDA components. Set timeout to 300+ seconds.

### Build and Test Commands

#### SSMProblems Package
```bash
cd SSMProblems
julia --project -e "using Pkg; Pkg.test()"
```
**Timing**: Takes ~20 seconds. Set timeout to 60+ seconds.

#### GeneralisedFilters Package  
```bash
cd GeneralisedFilters
julia --project -e "using Pkg; Pkg.test()"
```
**NEVER CANCEL**: Tests take 60+ seconds due to GPU/CUDA components. Set timeout to 180+ seconds.

### Code Formatting
```bash
# Format both packages (run from repository root)
julia -e "using Pkg; Pkg.add(PackageSpec(name=\"JuliaFormatter\", uuid=\"98e50ef6-434e-11e9-1051-2b60c6c9e899\")); using JuliaFormatter; format(\"SSMProblems\"); format(\"GeneralisedFilters\")"
```
**Timing**: Takes ~60 seconds for setup and formatting. Always run before committing.

### Documentation Building
```bash
# Build SSMProblems documentation
cd SSMProblems/docs
julia --project -e "using Pkg; Pkg.develop(path=\"..\"); Pkg.instantiate()"
julia --project make.jl

# Build GeneralisedFilters documentation  
cd ../../GeneralisedFilters/docs
julia --project -e "using Pkg; Pkg.develop(path=\"..\"); Pkg.develop(path=\"../../SSMProblems\"); Pkg.instantiate()"
julia --project make.jl
```
**NEVER CANCEL**: Documentation builds take 10+ minutes due to example processing with CUDA components. Set timeout to 900+ seconds.

## Validation Procedures

### Manual Testing After Changes
Always perform these validation steps after making changes:

#### Basic Functionality Test
```bash
# Test SSMProblems can be loaded and basic functionality works
cd SSMProblems
julia --project -e "using SSMProblems; using Distributions; println(\"SSMProblems loaded successfully\")"

# Test GeneralisedFilters integration
cd ../GeneralisedFilters  
julia --project -e "using GeneralisedFilters, SSMProblems; println(\"Integration test passed\")"
```

#### Complete User Workflow Test
Always test a complete workflow after making changes:
```bash
# Test basic state space model creation and usage
cd SSMProblems
julia --project -e "
using SSMProblems, Distributions
# Create simple prior
struct TestPrior <: StatePrior
    μ::Float64
    σ²::Float64
end
SSMProblems.distribution(p::TestPrior; kwargs...) = Normal(p.μ, sqrt(p.σ²))
# Create simple dynamics
struct TestDynamics <: LatentDynamics end
SSMProblems.distribution(::TestDynamics, ::Int, state::Float64; kwargs...) = Normal(state, 0.1)
# Create observation process  
struct TestObs <: ObservationProcess end
SSMProblems.distribution(::TestObs, ::Int, state::Float64; kwargs...) = Normal(state, 0.2)
# Test model creation
prior = TestPrior(0.0, 1.0)
model = StateSpaceModel(prior, TestDynamics(), TestObs())
println(\"Basic SSM workflow test passed\")
"

# Test filtering algorithms work with SSM models
cd ../GeneralisedFilters
julia --project -e "
using GeneralisedFilters, SSMProblems, Distributions, Random
# Use example from the codebase
rng = MersenneTwister(1234)
model = GeneralisedFilters.GFTest.create_linear_gaussian_model(rng, 2, 2)
_, _, ys = sample(rng, model, 5)
state, ll = GeneralisedFilters.filter(rng, model, KalmanFilter(), ys)
println(\"Filtering workflow test passed: loglikelihood = \$ll\")
"
```

#### CI Validation Commands
Always run these before committing to ensure CI will pass:
```bash
# Format check (from repository root)
julia -e "using JuliaFormatter; format(\"SSMProblems\"; verbose=false); format(\"GeneralisedFilters\"; verbose=false)"

# Test both packages
cd SSMProblems && julia --project -e "using Pkg; Pkg.test()"
cd ../GeneralisedFilters && julia --project -e "using Pkg; Pkg.test()"
```

## Key Repository Structure

### Important Files and Directories
```
.
├── .JuliaFormatter.toml          # Blue style formatting config
├── .github/workflows/            # CI/CD workflows
│   ├── Documentation.yml         # Docs building 
│   ├── Format.yml                # Code formatting checks
│   └── ...
├── .buildkite/pipeline.yml       # GPU testing configuration
├── SSMProblems/                  # Core package
│   ├── Project.toml              # Package dependencies
│   ├── src/                      # Source code
│   ├── test/                     # Test suite
│   ├── docs/                     # Documentation
│   └── examples/                 # Example projects
└── GeneralisedFilters/           # Filtering algorithms package
    ├── Project.toml              # Package dependencies (includes CUDA)
    ├── src/                      # Source code  
    ├── test/                     # Test suite (includes GPU tests)
    ├── docs/                     # Documentation
    └── examples/                 # Example projects
```

### Working with Examples
Both packages include examples that are processed during documentation builds:
```bash
# Examples are in subdirectories with their own Project.toml files
ls SSMProblems/examples/          # kalman-filter/
ls GeneralisedFilters/examples/   # demo-example/, trend-inflation/
```

## Development Workflows

### Making Changes to SSMProblems
1. Work in the `SSMProblems/` directory
2. Test changes: `julia --project -e "using Pkg; Pkg.test()"`
3. If changing interfaces, test impact on GeneralisedFilters:
   ```bash
   cd ../GeneralisedFilters
   julia --project -e "using Pkg; Pkg.develop(path=\"../SSMProblems\"); Pkg.test()"
   ```

### Making Changes to GeneralisedFilters
1. Work in the `GeneralisedFilters/` directory  
2. Test changes: `julia --project -e "using Pkg; Pkg.test()"`
3. GeneralisedFilters automatically uses local SSMProblems via `Pkg.develop(path="../SSMProblems")`

### Documentation Changes
- Documentation source files are in `*/docs/src/`
- Examples are processed using Literate.jl during doc builds
- Always rebuild docs after changes: `julia docs/make.jl`

## Common Issues and Solutions

### Package Dependency Issues
If you see package resolution errors:
```bash
# Clear package cache and reinstall
julia -e "using Pkg; Pkg.gc(); Pkg.resolve()"
cd SSMProblems && julia --project -e "using Pkg; Pkg.instantiate()"
cd ../GeneralisedFilters && julia --project -e "using Pkg; Pkg.instantiate()"
```

### CUDA/GPU Issues
GeneralisedFilters includes CUDA support. If tests fail with GPU errors:
- Tests automatically skip GPU functionality if CUDA unavailable
- GPU tests run on Buildkite CI with dedicated GPU agents
- Local development doesn't require GPU hardware

### Formatting Issues
The repository uses Julia Blue formatting style:
```bash
# Check if formatting is needed
julia -e "using JuliaFormatter; format(\".\"; verbose=true)"
```

## Time Expectations

Set appropriate timeouts for these operations (**measured on Julia 1.11.6**):
- **SSMProblems dependency install**: 60 seconds
- **GeneralisedFilters dependency install**: 300 seconds (**NEVER CANCEL**)
- **SSMProblems tests**: 60 seconds  
- **GeneralisedFilters tests**: 180 seconds (**NEVER CANCEL**)
- **Documentation builds**: 900 seconds (**NEVER CANCEL**)
- **Formatting**: 120 seconds

## Hardware Requirements

- **SSMProblems**: Standard Julia environment, no special requirements
- **GeneralisedFilters**: Includes CUDA support for GPU acceleration
  - GPU hardware is optional for development
  - Tests automatically skip GPU functionality if CUDA unavailable
  - GPU tests run on Buildkite CI with dedicated GPU agents

## CI/CD Information

- **GitHub Actions**: Handles documentation building and formatting checks
- **Buildkite**: Handles GPU testing with CUDA-enabled agents
- **Auto-formatting**: PRs are automatically checked for formatting compliance
- **Documentation**: Auto-deployed to GitHub Pages on merge to main

Always run local validation before pushing to ensure CI passes.

## Common Command Outputs

The following are outputs from frequently run commands. Reference them instead of running bash commands unnecessarily to save time.

### Repository Structure
```bash
ls -la
# Output:
.JuliaFormatter.toml
.buildkite/
.git/
.github/
.gitignore
GeneralisedFilters/
HISTORY.md
LICENSE
README.md
SSMProblems/
research/
```

### Package Status
```bash
cd SSMProblems && julia --project -e "using Pkg; Pkg.status()"
# Output:
Project SSMProblems v0.6.0
Status `~/SSMProblems/Project.toml`
  [80f14c24] AbstractMCMC v5.7.2
  [31c24e10] Distributions v0.25.120
  [9a3f8284] Random
```

### Test Summary Example
```bash
cd SSMProblems && julia --project -e "using Pkg; Pkg.test()"
# Typical output:
Test Summary:      | Pass  Total  Time
Forward Simulation |    4      4  1.0s
Test Summary: | Pass  Total   Time
Aqua.jl QA    |    7      7  18.1s
     Testing SSMProblems tests passed
```