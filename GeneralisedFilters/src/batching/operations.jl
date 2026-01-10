import PDMats: X_A_Xt
import LinearAlgebra: norm

# =============================================================================
# GEMM-Compatible Types
# =============================================================================

# Type aliases for StructArray-wrapped matrices
const BatchedAdjoint{T,M} = StructArray{
    Adjoint{T,CuArray{T,2,M}},1,@NamedTuple{parent::BatchedCuMatrix{T,M}}
}
const BatchedTranspose{T,M} = StructArray{
    Transpose{T,CuArray{T,2,M}},1,@NamedTuple{parent::BatchedCuMatrix{T,M}}
}
const SharedAdjoint{T,M} = StructArray{
    Adjoint{T,CuArray{T,2,M}},1,@NamedTuple{parent::SharedCuMatrix{T,M}}
}
const SharedTranspose{T,M} = StructArray{
    Transpose{T,CuArray{T,2,M}},1,@NamedTuple{parent::SharedCuMatrix{T,M}}
}

# Union of all GEMM-compatible matrix types
const GEMMCompatibleMatrix{T} = Union{
    BatchedCuMatrix{T},
    SharedCuMatrix{T},
    BatchedAdjoint{T},
    BatchedTranspose{T},
    SharedAdjoint{T},
    SharedTranspose{T},
}

# trans_flag: returns BLAS transpose flag for each type
trans_flag(::BatchedCuMatrix{T}) where {T} = 'N'
trans_flag(::SharedCuMatrix{T}) where {T} = 'N'
trans_flag(::BatchedAdjoint{T}) where {T} = T <: Real ? 'T' : 'C'
trans_flag(::BatchedTranspose{T}) where {T} = 'T'
trans_flag(::SharedAdjoint{T}) where {T} = T <: Real ? 'T' : 'C'
trans_flag(::SharedTranspose{T}) where {T} = 'T'

# gemm_data: extracts the underlying BatchedCuMatrix/SharedCuMatrix for GEMM
gemm_data(A::BatchedCuMatrix) = A
gemm_data(A::SharedCuMatrix) = A
gemm_data(A::BatchedAdjoint) = A.parent
gemm_data(A::BatchedTranspose) = A.parent
gemm_data(A::SharedAdjoint) = A.parent
gemm_data(A::SharedTranspose) = A.parent

# inner_size_for_blas for wrapped types (delegates to underlying data)
inner_size_for_blas(A::BatchedAdjoint) = inner_size_for_blas(A.parent)
inner_size_for_blas(A::BatchedTranspose) = inner_size_for_blas(A.parent)
inner_size_for_blas(A::SharedAdjoint) = inner_size_for_blas(A.parent)
inner_size_for_blas(A::SharedTranspose) = inner_size_for_blas(A.parent)

# batch_size for wrapped types
batch_size(A::BatchedAdjoint) = batch_size(A.parent)
batch_size(A::BatchedTranspose) = batch_size(A.parent)
batch_size(A::SharedAdjoint) = batch_size(A.parent)
batch_size(A::SharedTranspose) = batch_size(A.parent)

# TODO: For nested wrappers (e.g., Adjoint{LowerTriangular{...}}), we should
# materialize the inner wrapper first before extracting. For now, we only
# support single-level Adjoint/Transpose wrappers for efficient GEMM dispatch.

# =============================================================================
# Matrix Multiply Broadcasting
# =============================================================================

function broadcasted(
    ::typeof(*), A::GEMMCompatibleMatrix{T}, B::GEMMCompatibleMatrix{T}
) where {T}
    transA = trans_flag(A)
    transB = trans_flag(B)

    A_inner = inner_size_for_blas(A)
    B_inner = inner_size_for_blas(B)

    m = transA == 'N' ? A_inner[1] : A_inner[2]
    n = transB == 'N' ? B_inner[2] : B_inner[1]
    N = get_batch_size(A, B)

    C_data = CuArray{T}(undef, m, n, N)
    C = BatchedCuMatrix(C_data)

    gemm_batched!(transA, transB, one(T), gemm_data(A), gemm_data(B), zero(T), C)
    return C
end

# Multi-argument multiply
function broadcasted(
    ::typeof(*),
    A::GEMMCompatibleMatrix{T},
    B::GEMMCompatibleMatrix{T},
    C::GEMMCompatibleMatrix{T},
    rest::GEMMCompatibleMatrix{T}...,
) where {T}
    result = broadcasted(*, A, B)
    result = broadcasted(*, result, C)
    for R in rest
        result = broadcasted(*, result, R)
    end
    return result
end

# =============================================================================
# Matrix-Vector Multiply Broadcasting
# =============================================================================

function broadcasted(
    ::typeof(*),
    A::Union{BatchedCuMatrix{T},SharedCuMatrix{T}},
    x::Union{BatchedCuVector{T},SharedCuVector{T}},
) where {T}
    transA = trans_flag(A)
    A_inner = inner_size_for_blas(A)
    m = transA == 'N' ? A_inner[1] : A_inner[2]
    N = get_batch_size(A, x)

    y_data = CuArray{T}(undef, m, N)
    y = BatchedCuVector(y_data)

    gemv_batched!(transA, one(T), A, x, zero(T), y)

    return y
end

# =============================================================================
# Identity Minus Matrix (I - A) Custom Kernel
# =============================================================================

function identity_minus_kernel!(C, A, m, n)
    batch_idx = blockIdx().z
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    if i <= m && j <= n
        if i == j
            @inbounds C[i, j, batch_idx] = one(eltype(C)) - A[i, j, batch_idx]
        else
            @inbounds C[i, j, batch_idx] = -A[i, j, batch_idx]
        end
    end
    return nothing
end

function identity_minus_batched!(C::CuArray{T,3}, A::CuArray{T,3}) where {T}
    m, n, N = size(A)
    threads = (16, 16)
    blocks = (cld(m, 16), cld(n, 16), N)
    @cuda threads = threads blocks = blocks identity_minus_kernel!(C, A, m, n)
    return C
end

function broadcasted(
    ::typeof(-), ::Base.RefValue{UniformScaling{Bool}}, A::BatchedCuMatrix{T}
) where {T}
    C = CuArray{T}(undef, size(A.data))
    identity_minus_batched!(C, A.data)
    return BatchedCuMatrix(C)
end

# =============================================================================
# Matrix Plus Scaled Identity (A + λI) Custom Kernel
# =============================================================================

function plus_scaled_identity_kernel!(C, A, λ, m, n)
    batch_idx = blockIdx().z
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    if i <= m && j <= n
        if i == j
            @inbounds C[i, j, batch_idx] = A[i, j, batch_idx] + λ
        else
            @inbounds C[i, j, batch_idx] = A[i, j, batch_idx]
        end
    end
    return nothing
end

function plus_scaled_identity_batched!(C::CuArray{T,3}, A::CuArray{T,3}, λ::T) where {T}
    m, n, N = size(A)
    threads = (16, 16)
    blocks = (cld(m, 16), cld(n, 16), N)
    @cuda threads = threads blocks = blocks plus_scaled_identity_kernel!(C, A, λ, m, n)
    return C
end

# A + Ref(I) where I is unscaled identity
function broadcasted(
    ::typeof(+), A::BatchedCuMatrix{T}, ::Base.RefValue{UniformScaling{Bool}}
) where {T}
    C = CuArray{T}(undef, size(A.data))
    plus_scaled_identity_batched!(C, A.data, one(T))
    return BatchedCuMatrix(C)
end

function broadcasted(
    ::typeof(+), ::Base.RefValue{UniformScaling{Bool}}, A::BatchedCuMatrix{T}
) where {T}
    C = CuArray{T}(undef, size(A.data))
    plus_scaled_identity_batched!(C, A.data, one(T))
    return BatchedCuMatrix(C)
end

# A + λI where λI is a scaled UniformScaling
function broadcasted(
    ::typeof(+), A::BatchedCuMatrix{T}, J::Base.RefValue{<:UniformScaling{T}}
) where {T}
    C = CuArray{T}(undef, size(A.data))
    plus_scaled_identity_batched!(C, A.data, J[].λ)
    return BatchedCuMatrix(C)
end

function broadcasted(
    ::typeof(+), J::Base.RefValue{<:UniformScaling{T}}, A::BatchedCuMatrix{T}
) where {T}
    C = CuArray{T}(undef, size(A.data))
    plus_scaled_identity_batched!(C, A.data, J[].λ)
    return BatchedCuMatrix(C)
end

# =============================================================================
# PDMat Broadcasting
# =============================================================================

# HACK: PDMat is a constructor so will use
# `broadcasted(::Type{W}, args::Union{BatchedCuMatrix, BatchedCuVector, SharedCuMatrix, SharedCuVector, StructArray}...) where W`
# rather than the desired recursive broadcast to
# `PDMat(mat::AbstractMatrix) = PDMat(mat, cholesky(mat))`
# This method hardcodes a manual override for BatchedCuMatrix inputs. This should be replaced
# by a more general solution in the future.
function broadcasted(::Type{PDMat}, A::BatchedCuMatrix{T}) where {T}
    chol = cholesky.(A)
    return PDMat.(A, chol)
end

# HACK: Addition with PDMat extracts .mat field. Should be replaced by automatic
# materialization of PDMat to BatchedCuMatrix in the future.
function broadcasted(
    ::typeof(+), A::BatchedCuMatrix{T}, P::StructArray{<:PDMat{T}}
) where {T}
    return broadcasted(+, A, P.mat)
end

function broadcasted(
    ::typeof(+), P::StructArray{<:PDMat{T}}, A::BatchedCuMatrix{T}
) where {T}
    return broadcasted(+, P.mat, A)
end

# A / S where S is PDMat: computes A * inv(S)
# potrs solves S * X = B, so we solve S * X = A' and transpose back
function broadcasted(
    ::typeof(/), A::BatchedCuMatrix{T}, S::StructArray{<:PDMat{T}}
) where {T}
    L = S.chol.factors.data

    # Transpose A: potrs solves S*X = B, we want A*inv(S) = (inv(S)*A')'
    At = BatchedCuMatrix(permutedims(A.data, (2, 1, 3)))

    # Solve S * X = A' in-place (result stored in At)
    potrs_batched!('L', L, At)

    # Transpose back
    return BatchedCuMatrix(permutedims(At.data, (2, 1, 3)))
end

# =============================================================================
# Quadratic Form Broadcasting
# =============================================================================

# HACK: treat this as two GEMMs for now
function broadcasted(
    ::typeof(X_A_Xt),
    A::Union{BatchedCuMatrix{T},SharedCuMatrix{T}},
    X::Union{BatchedCuMatrix{T},SharedCuMatrix{T}},
) where {T}
    temp = broadcasted(*, X, A)
    Xt = broadcasted(adjoint, X)
    return broadcasted(*, temp, Xt)
end

# X_A_Xt for StructArray{PDMat}: X * P * X' where P = L * L'
# Computed as (X * L) * (X * L)' using TRMM and SYRK
# HACK: this function should dispatch to specialised `*` for triangular types but this is
# not yet implemented
function broadcasted(
    ::typeof(X_A_Xt),
    P::StructArray{<:PDMat{T}},
    X::Union{BatchedCuMatrix{T},SharedCuMatrix{T}},
) where {T}
    # P.chol.factors is StructArray{LowerTriangular}, .data is the BatchedCuMatrix
    L = P.chol.factors.data
    N = get_batch_size(L, X)
    out_dim = inner_size_for_blas(X)[1]

    # Copy X for in-place TRMM
    XL = if X isa SharedCuMatrix
        BatchedCuMatrix(repeat(reshape(X.data, size(X.data)..., 1), 1, 1, N))
    else
        BatchedCuMatrix(copy(X.data))
    end

    # XL = X * L using TRMM (side='R', uplo='L', no transpose, non-unit diagonal)
    trmm_batched!('R', 'L', 'N', 'N', one(T), L, XL)

    # result = XL * XL' using SYRK (fills lower triangle only)
    result = BatchedCuMatrix(CuArray{T}(undef, out_dim, out_dim, N))
    syrk_batched!('L', 'N', one(T), XL, zero(T), result)

    # Copy lower triangle to upper for full symmetric matrix
    symmetrize_lower!(result)

    return result
end

# =============================================================================
# Batched norm
# =============================================================================

# Compute 2-norm for each vector in the batch, returns a CuVector of scalars
function broadcasted(::typeof(norm), v::BatchedCuVector{T}) where {T}
    # v.data is D×N, compute norm of each column
    return vec(sqrt.(sum(abs2, v.data; dims=1)))
end

# =============================================================================
# Batched ifelse (for conditional selection with batched conditions)
# =============================================================================

# Select entire vectors: x[:,j] if cond[j], else y[:,j]
function broadcasted(
    ::typeof(ifelse), cond::CuVector{Bool}, x::BatchedCuVector{T}, y::BatchedCuVector{T}
) where {T}
    # cond is length N (one bool per batch element)
    # x.data and y.data are D×N
    mask = reshape(T.(cond), 1, :)  # 1×N mask for column selection
    result = mask .* x.data .+ (one(T) .- mask) .* y.data
    return BatchedCuVector(result)
end

# Select entire matrices: x[:,:,j] if cond[j], else y[:,:,j]
function broadcasted(
    ::typeof(ifelse), cond::CuVector{Bool}, x::BatchedCuMatrix{T}, y::BatchedCuMatrix{T}
) where {T}
    # cond is length N (one bool per batch element)
    # x.data and y.data are D×D×N
    mask = reshape(T.(cond), 1, 1, :)  # 1×1×N mask for batch selection
    result = mask .* x.data .+ (one(T) .- mask) .* y.data
    return BatchedCuMatrix(result)
end
