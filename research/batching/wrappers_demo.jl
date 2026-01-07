using GeneralisedFilters
using CUDA
using LinearAlgebra
using StructArrays
using Base.Broadcast: broadcasted
using Distributions
using PDMats

# =============================================================================
# Configuration
# =============================================================================

N = 10  # batch size
D = 4   # matrix dimension

println("Creating batched matrices...")
A = BatchedCuMatrix(CUDA.randn(Float32, D, D, N));

# =============================================================================
# Test 1: Basic wrapper creation
# =============================================================================

println("\n=== Test 1: Basic wrapper creation ===\n")

# Adjoint
A_adj = broadcasted(Adjoint, A);
println("Adjoint type: ", typeof(A_adj))
println("Adjoint eltype: ", eltype(A_adj))
println("First element type: ", typeof(A_adj[1]))

# Transpose
A_trans = broadcasted(Transpose, A);
println("\nTranspose type: ", typeof(A_trans))
println("Transpose eltype: ", eltype(A_trans))

# LowerTriangular
A_lower = broadcasted(LowerTriangular, A);
println("\nLowerTriangular type: ", typeof(A_lower))
println("LowerTriangular eltype: ", eltype(A_lower))

# UpperTriangular
A_upper = broadcasted(UpperTriangular, A);
println("\nUpperTriangular type: ", typeof(A_upper))
println("UpperTriangular eltype: ", eltype(A_upper))

# =============================================================================
# Test 2: Function form redirects
# =============================================================================

println("\n=== Test 2: Function form redirects ===\n")

A_adj2 = broadcasted(adjoint, A);
println("adjoint redirect type: ", typeof(A_adj2))
println("Types match: ", typeof(A_adj) == typeof(A_adj2))

A_trans2 = broadcasted(transpose, A);
println("\ntranspose redirect type: ", typeof(A_trans2))
println("Types match: ", typeof(A_trans) == typeof(A_trans2))

# =============================================================================
# Test 3: Nested wrappers
# =============================================================================

println("\n=== Test 3: Nested wrappers ===\n")

# Adjoint of LowerTriangular
A_lower_adj = broadcasted(Adjoint, A_lower);
println("Adjoint(LowerTriangular) type: ", typeof(A_lower_adj))
println("Adjoint(LowerTriangular) eltype: ", eltype(A_lower_adj))

# =============================================================================
# Test 4: Verify element access
# =============================================================================

println("\n=== Test 4: Element access verification ===\n")

# Get first element of each wrapped array
println("A[1] type: ", typeof(A[1]))
println("A_adj[1] type: ", typeof(A_adj[1]))
println("A_lower[1] type: ", typeof(A_lower[1]))

# Check values match
A_cpu = Array(A[1]);
A_adj_cpu = Array(parent(A_adj[1]));
println("\nValues match (A vs parent of A_adj): ", A_cpu ≈ A_adj_cpu)

# =============================================================================
# Test 5: SharedCuMatrix wrappers
# =============================================================================

println("\n=== Test 5: SharedCuMatrix wrappers ===\n")

S = SharedCuMatrix(CUDA.randn(Float32, D, D));
S_adj = broadcasted(Adjoint, S);
println("SharedCuMatrix adjoint type: ", typeof(S_adj))
println("SharedCuMatrix adjoint eltype: ", eltype(S_adj))

# =============================================================================
# Test 6: Cholesky
# =============================================================================

println("\n=== Test 6: Batched Cholesky ===\n")

# Create positive definite matrices: B = A * A'
B = A .* broadcasted(adjoint, A);
println("B type: ", typeof(B))

chol_result = cholesky.(B);
println("Cholesky result type: ", typeof(chol_result))
println("Cholesky eltype: ", eltype(chol_result))

# Check fields
println("\nCholesky fields:")
println("  factors type: ", typeof(chol_result.factors))
println("  uplo type: ", typeof(chol_result.uplo))
println("  info type: ", typeof(chol_result.info))

# Access individual element
# SKIP: requires scalar indexing into info
# println("\nFirst element:")
# println("  chol_result[1] type: ", typeof(chol_result[1]))

# =============================================================================
# Test 7: PDMat wrapper
# =============================================================================

println("\n=== Test 7: PDMat wrapper ===\n")

P = broadcasted(PDMat, B, chol_result);
println("PDMat result type: ", typeof(P))
println("PDMat eltype: ", eltype(P))

println("\nPDMat fields:")
# println("  dim type: ", typeof(P.dim))  # SKIP: not a real field
println("  mat type: ", typeof(P.mat))
println("  chol type: ", typeof(P.chol))

# =============================================================================
# Test 8: MvNormal wrapper
# =============================================================================

println("\n=== Test 8: MvNormal wrapper ===\n")
μ = BatchedCuVector(CUDA.randn(Float32, D, N));
G = broadcasted(MvNormal, μ, P);
println("MvNormal type: ", typeof(G))
println("MvNormal eltype: ", eltype(G))
