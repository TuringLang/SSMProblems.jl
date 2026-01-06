using Magma
using Magma.LibMagma

# =============================================================================
# Trivial Wrappers (reductions and elementwise operations)
# =============================================================================

function broadcasted(
    ::typeof(+),
    A::Union{BatchedCuMatrix{T},SharedCuMatrix{T}},
    B::Union{BatchedCuMatrix{T},SharedCuMatrix{T}},
) where {T}
    if is_shared(A) && is_shared(B)
        return SharedCuMatrix(A.data .+ B.data)
    else
        return BatchedCuMatrix(A.data .+ B.data)
    end
end

function broadcasted(
    ::typeof(+),
    a::Union{BatchedCuVector{T},SharedCuVector{T}},
    b::Union{BatchedCuVector{T},SharedCuVector{T}},
) where {T}
    if is_shared(a) && is_shared(b)
        return SharedCuVector(a.data .+ b.data)
    else
        return BatchedCuVector(a.data .+ b.data)
    end
end

function broadcasted(
    ::typeof(-),
    A::Union{BatchedCuMatrix{T},SharedCuMatrix{T}},
    B::Union{BatchedCuMatrix{T},SharedCuMatrix{T}},
) where {T}
    if is_shared(A) && is_shared(B)
        return SharedCuMatrix(A.data .- B.data)
    else
        return BatchedCuMatrix(A.data .- B.data)
    end
end

function broadcasted(
    ::typeof(-),
    a::Union{BatchedCuVector{T},SharedCuVector{T}},
    b::Union{BatchedCuVector{T},SharedCuVector{T}},
) where {T}
    if is_shared(a) && is_shared(b)
        return SharedCuVector(a.data .- b.data)
    else
        return BatchedCuVector(a.data .- b.data)
    end
end

# =============================================================================
# MAGMA Constants Conversion
# =============================================================================

function magma_trans(c::Char)
    if c == 'N'
        return LibMagma.MagmaNoTrans
    elseif c == 'T'
        return LibMagma.MagmaTrans
    elseif c == 'C'
        return LibMagma.MagmaConjTrans
    else
        error("Unknown transpose char: $c")
    end
end

function magma_uplo(c::Char)
    if c == 'L'
        return LibMagma.MagmaLower
    elseif c == 'U'
        return LibMagma.MagmaUpper
    else
        error("Unknown uplo char: $c")
    end
end

function magma_side(c::Char)
    if c == 'L'
        return LibMagma.MagmaLeft
    elseif c == 'R'
        return LibMagma.MagmaRight
    else
        error("Unknown side char: $c")
    end
end

function magma_diag(c::Char)
    if c == 'N'
        return LibMagma.MagmaNonUnit
    elseif c == 'U'
        return LibMagma.MagmaUnit
    else
        error("Unknown diag char: $c")
    end
end

# =============================================================================
# MAGMA Operations
# =============================================================================

function gemm_batched!(
    transA::Char,
    transB::Char,
    alpha::Float32,
    A::Union{BatchedCuMatrix{Float32},SharedCuMatrix{Float32}},
    B::Union{BatchedCuMatrix{Float32},SharedCuMatrix{Float32}},
    beta::Float32,
    C::BatchedCuMatrix{Float32},
)
    N = batch_size(C)
    m, n = size(C.data, 1), size(C.data, 2)
    k = transA == 'N' ? size(unwrap_data(A), 2) : size(unwrap_data(A), 1)

    dA = A isa BatchedCuMatrix ? create_pointer_array(A) : create_pointer_array(A, N)
    dB = B isa BatchedCuMatrix ? create_pointer_array(B) : create_pointer_array(B, N)
    dC = create_pointer_array(C)

    ldda = size(unwrap_data(A), 1)
    lddb = size(unwrap_data(B), 1)
    lddc = m

    CUDA.synchronize()
    queue_ptr = Ref{LibMagma.magma_queue_t}()
    LibMagma.magma_queue_create_internal(0, queue_ptr, C_NULL, C_NULL, 0)
    LibMagma.magma_sgemm_batched(
        magma_trans(transA),
        magma_trans(transB),
        m,
        n,
        k,
        alpha,
        dA,
        ldda,
        dB,
        lddb,
        beta,
        dC,
        lddc,
        N,
        queue_ptr[],
    )
    LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
    LibMagma.magma_queue_destroy_internal(queue_ptr[], C_NULL, C_NULL, 0)

    CUDA.unsafe_free!(dA)
    CUDA.unsafe_free!(dB)
    CUDA.unsafe_free!(dC)

    return C
end

# =============================================================================
# Part 3b: Batched GEMM Small Square Wrapper (for D < 32)
# =============================================================================

function gemm_batched_smallsq!(
    transA::Char,
    transB::Char,
    alpha::Float32,
    A::Union{BatchedCuMatrix{Float32},SharedCuMatrix{Float32}},
    B::Union{BatchedCuMatrix{Float32},SharedCuMatrix{Float32}},
    beta::Float32,
    C::BatchedCuMatrix{Float32},
)
    N = batch_size(C)
    m, n = size(C.data, 1), size(C.data, 2)
    k = transA == 'N' ? size(unwrap_data(A), 2) : size(unwrap_data(A), 1)

    dA = A isa BatchedCuMatrix ? create_pointer_array(A) : create_pointer_array(A, N)
    dB = B isa BatchedCuMatrix ? create_pointer_array(B) : create_pointer_array(B, N)
    dC = create_pointer_array(C)

    ldda = size(unwrap_data(A), 1)
    lddb = size(unwrap_data(B), 1)
    lddc = m

    CUDA.synchronize()
    queue_ptr = Ref{LibMagma.magma_queue_t}()
    LibMagma.magma_queue_create_internal(0, queue_ptr, C_NULL, C_NULL, 0)
    LibMagma.magmablas_sgemm_batched_smallsq(
        magma_trans(transA),
        magma_trans(transB),
        m,
        n,
        k,
        alpha,
        dA,
        0,      # ai
        0,      # aj
        ldda,
        dB,
        0,      # bi
        0,      # bj
        lddb,
        beta,
        dC,
        0,      # ci
        0,      # cj
        lddc,
        N,
        queue_ptr[],
    )
    LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
    LibMagma.magma_queue_destroy_internal(queue_ptr[], C_NULL, C_NULL, 0)

    CUDA.unsafe_free!(dA)
    CUDA.unsafe_free!(dB)
    CUDA.unsafe_free!(dC)

    return C
end

function gemm_batched_smallsq!(
    transA::Char,
    transB::Char,
    alpha::Float64,
    A::Union{BatchedCuMatrix{Float64},SharedCuMatrix{Float64}},
    B::Union{BatchedCuMatrix{Float64},SharedCuMatrix{Float64}},
    beta::Float64,
    C::BatchedCuMatrix{Float64},
)
    N = batch_size(C)
    m, n = size(C.data, 1), size(C.data, 2)
    k = transA == 'N' ? size(unwrap_data(A), 2) : size(unwrap_data(A), 1)

    dA = A isa BatchedCuMatrix ? create_pointer_array(A) : create_pointer_array(A, N)
    dB = B isa BatchedCuMatrix ? create_pointer_array(B) : create_pointer_array(B, N)
    dC = create_pointer_array(C)

    ldda = size(unwrap_data(A), 1)
    lddb = size(unwrap_data(B), 1)
    lddc = m

    CUDA.synchronize()
    queue_ptr = Ref{LibMagma.magma_queue_t}()
    LibMagma.magma_queue_create_internal(0, queue_ptr, C_NULL, C_NULL, 0)
    LibMagma.magmablas_dgemm_batched_smallsq(
        magma_trans(transA),
        magma_trans(transB),
        m,
        n,
        k,
        alpha,
        dA,
        0,      # ai
        0,      # aj
        ldda,
        dB,
        0,      # bi
        0,      # bj
        lddb,
        beta,
        dC,
        0,      # ci
        0,      # cj
        lddc,
        N,
        queue_ptr[],
    )
    LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
    LibMagma.magma_queue_destroy_internal(queue_ptr[], C_NULL, C_NULL, 0)

    CUDA.unsafe_free!(dA)
    CUDA.unsafe_free!(dB)
    CUDA.unsafe_free!(dC)

    return C
end

function gemv_batched!(
    transA::Char,
    alpha::Float32,
    A::Union{BatchedCuMatrix{Float32},SharedCuMatrix{Float32}},
    x::Union{BatchedCuVector{Float32},SharedCuVector{Float32}},
    beta::Float32,
    y::BatchedCuVector{Float32},
)
    N = batch_size(y)
    m, n = size(unwrap_data(A), 1), size(unwrap_data(A), 2)

    dA = A isa BatchedCuMatrix ? create_pointer_array(A) : create_pointer_array(A, N)
    dx = if x isa BatchedCuVector
        create_pointer_array_vector(x)
    else
        create_pointer_array_vector(x, N)
    end
    dy = create_pointer_array_vector(y)

    ldda = m
    incx = 1
    incy = 1

    CUDA.synchronize()
    queue_ptr = Ref{LibMagma.magma_queue_t}()
    LibMagma.magma_queue_create_internal(0, queue_ptr, C_NULL, C_NULL, 0)
    LibMagma.magmablas_sgemv_batched(
        magma_trans(transA), m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, N, queue_ptr[]
    )
    LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
    LibMagma.magma_queue_destroy_internal(queue_ptr[], C_NULL, C_NULL, 0)

    CUDA.unsafe_free!(dA)
    CUDA.unsafe_free!(dx)
    CUDA.unsafe_free!(dy)

    return y
end

function gemv_batched!(
    transA::Char,
    alpha::Float64,
    A::Union{BatchedCuMatrix{Float64},SharedCuMatrix{Float64}},
    x::Union{BatchedCuVector{Float64},SharedCuVector{Float64}},
    beta::Float64,
    y::BatchedCuVector{Float64},
)
    N = batch_size(y)
    m, n = size(unwrap_data(A), 1), size(unwrap_data(A), 2)

    dA = A isa BatchedCuMatrix ? create_pointer_array(A) : create_pointer_array(A, N)
    dx = if x isa BatchedCuVector
        create_pointer_array_vector(x)
    else
        create_pointer_array_vector(x, N)
    end
    dy = create_pointer_array_vector(y)

    ldda = m
    incx = 1
    incy = 1

    CUDA.synchronize()
    queue_ptr = Ref{LibMagma.magma_queue_t}()
    LibMagma.magma_queue_create_internal(0, queue_ptr, C_NULL, C_NULL, 0)
    LibMagma.magmablas_dgemv_batched(
        magma_trans(transA), m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, N, queue_ptr[]
    )
    LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
    LibMagma.magma_queue_destroy_internal(queue_ptr[], C_NULL, C_NULL, 0)

    CUDA.unsafe_free!(dA)
    CUDA.unsafe_free!(dx)
    CUDA.unsafe_free!(dy)

    return y
end

function gemv_batched_smallsq!(
    transA::Char,
    alpha::Float32,
    A::Union{BatchedCuMatrix{Float32},SharedCuMatrix{Float32}},
    x::Union{BatchedCuVector{Float32},SharedCuVector{Float32}},
    beta::Float32,
    y::BatchedCuVector{Float32},
)
    N = batch_size(y)
    n = size(unwrap_data(A), 1)

    dA = A isa BatchedCuMatrix ? create_pointer_array(A) : create_pointer_array(A, N)
    dx = if x isa BatchedCuVector
        create_pointer_array_vector(x)
    else
        create_pointer_array_vector(x, N)
    end
    dy = create_pointer_array_vector(y)

    ldda = n
    incx = 1
    incy = 1

    CUDA.synchronize()
    queue_ptr = Ref{LibMagma.magma_queue_t}()
    LibMagma.magma_queue_create_internal(0, queue_ptr, C_NULL, C_NULL, 0)
    LibMagma.magmablas_sgemv_batched_smallsq(
        magma_trans(transA), n, alpha, dA, ldda, dx, incx, beta, dy, incy, N, queue_ptr[]
    )
    LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
    LibMagma.magma_queue_destroy_internal(queue_ptr[], C_NULL, C_NULL, 0)

    CUDA.unsafe_free!(dA)
    CUDA.unsafe_free!(dx)
    CUDA.unsafe_free!(dy)

    return y
end

function potrf_batched!(uplo::Char, A::BatchedCuMatrix{Float32})
    N = batch_size(A)
    n = size(A.data, 1)
    lda = n

    dA = create_pointer_array(A)
    info_gpu = CUDA.zeros(Int64, N)

    CUDA.synchronize()
    queue_ptr = Ref{LibMagma.magma_queue_t}()
    LibMagma.magma_queue_create_internal(0, queue_ptr, C_NULL, C_NULL, 0)
    LibMagma.magma_spotrf_batched(
        magma_uplo(uplo), n, dA, lda, pointer(info_gpu), N, queue_ptr[]
    )
    LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
    LibMagma.magma_queue_destroy_internal(queue_ptr[], C_NULL, C_NULL, 0)

    CUDA.unsafe_free!(dA)

    factors = BatchedCuMatrix{Float32,LowerTriangular{Float32,CuMatrix{Float32}}}(A.data)
    return BatchedCholesky{Float32}(factors, info_gpu, uplo)
end

function potrs_batched!(
    uplo::Char, A::BatchedCuMatrix{Float32}, B::BatchedCuMatrix{Float32}
)
    N = batch_size(B)
    n = size(A.data, 1)
    nrhs = size(B.data, 2)

    dA = create_pointer_array(A)
    dB = create_pointer_array(B)

    ldda = n
    lddb = n

    CUDA.synchronize()
    queue_ptr = Ref{LibMagma.magma_queue_t}()
    LibMagma.magma_queue_create_internal(0, queue_ptr, C_NULL, C_NULL, 0)
    LibMagma.magma_spotrs_batched(
        magma_uplo(uplo), n, nrhs, dA, ldda, dB, lddb, N, queue_ptr[]
    )
    LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
    LibMagma.magma_queue_destroy_internal(queue_ptr[], C_NULL, C_NULL, 0)

    CUDA.unsafe_free!(dA)
    CUDA.unsafe_free!(dB)

    return B
end

function trsm_batched!(
    side::Char,
    uplo::Char,
    transA::Char,
    diag::Char,
    alpha::Float32,
    A::BatchedCuMatrix{Float32},
    B::BatchedCuMatrix{Float32},
)
    N = batch_size(B)
    m, n = size(B.data, 1), size(B.data, 2)

    dA = create_pointer_array(A)
    dB = create_pointer_array(B)

    ldda = size(A.data, 1)
    lddb = m

    CUDA.synchronize()
    queue_ptr = Ref{LibMagma.magma_queue_t}()
    LibMagma.magma_queue_create_internal(0, queue_ptr, C_NULL, C_NULL, 0)
    LibMagma.magmablas_strsm_batched(
        magma_side(side),
        magma_uplo(uplo),
        magma_trans(transA),
        magma_diag(diag),
        m,
        n,
        alpha,
        dA,
        ldda,
        dB,
        lddb,
        N,
        queue_ptr[],
    )
    LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
    LibMagma.magma_queue_destroy_internal(queue_ptr[], C_NULL, C_NULL, 0)

    CUDA.unsafe_free!(dA)
    CUDA.unsafe_free!(dB)

    return B
end

# =============================================================================
# Higher-level Cholesky Operations
# =============================================================================

function cholesky_batched(A::BatchedCuMatrix{T}) where {T}
    A_copy = BatchedCuMatrix(copy(A.data))
    return potrf_batched!('L', A_copy)
end

function pdmat_solve(S::BatchedPDMat{T}, B::BatchedCuMatrix{T}) where {T}
    L = S.chol.factors
    L_data = BatchedCuMatrix(L.data)

    B_copy = BatchedCuMatrix(copy(B.data))

    # Solve L*L'*X = B via two triangular solves:
    # 1. Solve L*Y = B (Y stored in B_copy)
    trsm_batched!('L', 'L', 'N', 'N', one(T), L_data, B_copy)
    # 2. Solve L'*X = Y (X stored in B_copy)
    trsm_batched!('L', 'L', 'T', 'N', one(T), L_data, B_copy)

    return B_copy
end

# =============================================================================
# Batched TRMM (Triangular Matrix Multiply)
# =============================================================================

function trmm_batched!(
    side::Char,
    uplo::Char,
    transA::Char,
    diag::Char,
    alpha::Float32,
    A::BatchedCuMatrix{Float32},
    B::BatchedCuMatrix{Float32},
)
    N = batch_size(B)
    m, n = size(B.data, 1), size(B.data, 2)

    dA = create_pointer_array(A)
    dB = create_pointer_array(B)

    ldda = size(A.data, 1)
    lddb = m

    CUDA.synchronize()
    queue_ptr = Ref{LibMagma.magma_queue_t}()
    LibMagma.magma_queue_create_internal(0, queue_ptr, C_NULL, C_NULL, 0)
    LibMagma.magmablas_strmm_batched(
        magma_side(side),
        magma_uplo(uplo),
        magma_trans(transA),
        magma_diag(diag),
        m,
        n,
        alpha,
        dA,
        ldda,
        dB,
        lddb,
        N,
        queue_ptr[],
    )
    LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
    LibMagma.magma_queue_destroy_internal(queue_ptr[], C_NULL, C_NULL, 0)

    CUDA.unsafe_free!(dA)
    CUDA.unsafe_free!(dB)

    return B
end

function trmm_batched!(
    side::Char,
    uplo::Char,
    transA::Char,
    diag::Char,
    alpha::Float64,
    A::BatchedCuMatrix{Float64},
    B::BatchedCuMatrix{Float64},
)
    N = batch_size(B)
    m, n = size(B.data, 1), size(B.data, 2)

    dA = create_pointer_array(A)
    dB = create_pointer_array(B)

    ldda = size(A.data, 1)
    lddb = m

    CUDA.synchronize()
    queue_ptr = Ref{LibMagma.magma_queue_t}()
    LibMagma.magma_queue_create_internal(0, queue_ptr, C_NULL, C_NULL, 0)
    LibMagma.magmablas_dtrmm_batched(
        magma_side(side),
        magma_uplo(uplo),
        magma_trans(transA),
        magma_diag(diag),
        m,
        n,
        alpha,
        dA,
        ldda,
        dB,
        lddb,
        N,
        queue_ptr[],
    )
    LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
    LibMagma.magma_queue_destroy_internal(queue_ptr[], C_NULL, C_NULL, 0)

    CUDA.unsafe_free!(dA)
    CUDA.unsafe_free!(dB)

    return B
end

# =============================================================================
# Batched SYRK (Symmetric Rank-K Update)
# =============================================================================

function syrk_batched!(
    uplo::Char,
    trans::Char,
    alpha::Float32,
    A::BatchedCuMatrix{Float32},
    beta::Float32,
    C::BatchedCuMatrix{Float32},
)
    N = batch_size(C)
    n = size(C.data, 1)
    k = trans == 'N' ? size(A.data, 2) : size(A.data, 1)

    dA = create_pointer_array(A)
    dC = create_pointer_array(C)

    ldda = size(A.data, 1)
    lddc = n

    CUDA.synchronize()
    queue_ptr = Ref{LibMagma.magma_queue_t}()
    LibMagma.magma_queue_create_internal(0, queue_ptr, C_NULL, C_NULL, 0)
    LibMagma.magmablas_ssyrk_batched(
        magma_uplo(uplo),
        magma_trans(trans),
        n,
        k,
        alpha,
        dA,
        ldda,
        beta,
        dC,
        lddc,
        N,
        queue_ptr[],
    )
    LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
    LibMagma.magma_queue_destroy_internal(queue_ptr[], C_NULL, C_NULL, 0)

    CUDA.unsafe_free!(dA)
    CUDA.unsafe_free!(dC)

    return C
end

function syrk_batched!(
    uplo::Char,
    trans::Char,
    alpha::Float64,
    A::BatchedCuMatrix{Float64},
    beta::Float64,
    C::BatchedCuMatrix{Float64},
)
    N = batch_size(C)
    n = size(C.data, 1)
    k = trans == 'N' ? size(A.data, 2) : size(A.data, 1)

    dA = create_pointer_array(A)
    dC = create_pointer_array(C)

    ldda = size(A.data, 1)
    lddc = n

    CUDA.synchronize()
    queue_ptr = Ref{LibMagma.magma_queue_t}()
    LibMagma.magma_queue_create_internal(0, queue_ptr, C_NULL, C_NULL, 0)
    LibMagma.magmablas_dsyrk_batched(
        magma_uplo(uplo),
        magma_trans(trans),
        n,
        k,
        alpha,
        dA,
        ldda,
        beta,
        dC,
        lddc,
        N,
        queue_ptr[],
    )
    LibMagma.magma_queue_sync_internal(queue_ptr[], C_NULL, C_NULL, 0)
    LibMagma.magma_queue_destroy_internal(queue_ptr[], C_NULL, C_NULL, 0)

    CUDA.unsafe_free!(dA)
    CUDA.unsafe_free!(dC)

    return C
end

# =============================================================================
# Symmetrize Lower Triangular Matrix
# =============================================================================

function symmetrize_lower_kernel!(A, n)
    batch_idx = blockIdx().z
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    if i <= n && j <= n && j > i
        @inbounds A[j, i, batch_idx] = A[i, j, batch_idx]
    end
    return nothing
end

function symmetrize_lower!(A::BatchedCuMatrix{T}) where {T}
    n = size(A.data, 1)
    N = size(A.data, 3)
    threads = (16, 16)
    blocks = (cld(n, 16), cld(n, 16), N)
    @cuda threads = threads blocks = blocks symmetrize_lower_kernel!(A.data, n)
    return A
end
