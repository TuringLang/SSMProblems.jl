import PDMats: X_A_Xt

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
# PDMat Broadcasting
# =============================================================================

# function broadcasted(::Type{PDMat}, A::BatchedCuMatrix{T,CuMatrix{T}}) where {T}
#     chol = cholesky_batched(A)
#     return BatchedPDMat{T}(chol)
# end

# function broadcasted(::typeof(\), S::BatchedPDMat{T}, A::BatchedCuMatrix{T}) where {T}
#     return pdmat_solve(S, A)
# end

# function broadcasted(::typeof(/), A::BatchedCuMatrix{T}, S::BatchedPDMat{T}) where {T}
#     # Need to actually transpose the data, not just wrap it
#     At_data = permutedims(A.data, (2, 1, 3))
#     At = BatchedCuMatrix(At_data)
#     result_t = pdmat_solve(S, At)
#     # Transpose back
#     result_data = permutedims(result_t.data, (2, 1, 3))
#     return BatchedCuMatrix(result_data)
# end

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

# X_A_Xt for BatchedPDMat: X * P * X' where P = L * L'
# Computed as (X * L) * (X * L)' using TRMM and SYRK
# function broadcasted(
#     ::typeof(X_A_Xt), P::BatchedPDMat{T}, X::Union{BatchedCuMatrix{T},SharedCuMatrix{T}}
# ) where {T}
#     L = P.chol.factors
#     N = get_batch_size(P, X)

#     X_inner = inner_size_for_blas(X)
#     m = X_inner[1]

#     # Copy X to XL (TRMM overwrites in-place)
#     XL_data = if X isa SharedCuMatrix
#         repeat(reshape(X.data, size(X.data, 1), size(X.data, 2), 1), 1, 1, N)
#     else
#         copy(X.data)
#     end
#     XL = BatchedCuMatrix(XL_data)

#     # XL = X * L using TRMM (side='R' for right multiply, uplo='L' for lower triangular)
#     L_data = BatchedCuMatrix(L.data)
#     trmm_batched!('R', 'L', 'N', 'N', one(T), L_data, XL)

#     # Result = XL * XL' using SYRK (fills lower triangle)
#     Result_data = CuArray{T}(undef, m, m, N)
#     Result = BatchedCuMatrix(Result_data)
#     syrk_batched!('L', 'N', one(T), XL, zero(T), Result)

#     # Symmetrize: copy lower triangle to upper
#     symmetrize_lower!(Result)

#     return Result
# end
