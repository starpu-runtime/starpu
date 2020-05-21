using LinearAlgebra.BLAS

function u11(sub11)
    nx = size(sub11, 1)
    ld = size(sub11, 1)

    for z in 0:nx-1
        lambda11::Float32 = sqrt(sub11[z+1,z+1])
        sub11[z+1,z+1] = lambda11
        if lambda11 == 0.0f0
            error("lamda11")
        end

        X = view(sub11, z+2:z+2+(nx-z-2), z+1)
        scal!(nx-z-1, 1.0f0/lambda11, X, 1)

        A = view(sub11, z+2:z+2+(nx-z-2), z+2:z+2+(nx-z-2))
        syr!('L', -1.0f0, X, A)
    end
end

function u21(sub11, sub21)
    trsm!('R', 'L', 'T', 'N', 1.0f0, sub11, sub21)
end

function u22(left, right, center)
    gemm!('N', 'T', -1.0f0, left, right, 1.0f0, center)
end

function get_block(mat :: Matrix{Float32}, m, n, nblocks)
    dim = size(mat, 1)
    if dim != size(mat,2)
        error("mat must be a square matrix")
    end
    if dim % nblocks != 0
        error("dim must be a multiple of nblocks")
    end

    stride = Int(dim/nblocks)

    return view(mat,
                m*stride+1:(m+1)*stride,
                n*stride+1:(n+1)*stride)
end

function cholesky(mat :: Matrix{Float32}, size, nblocks)
    for k in 0:nblocks-1
        sdatakk = get_block(mat, k, k, nblocks)
        u11(sdatakk)

        for m in k+1:nblocks-1
            sdatamk = get_block(mat, m, k, nblocks)
            u21(sdatakk, sdatamk)
        end

        for m in k+1:nblocks-1
            sdatamk = get_block(mat, m, k, nblocks)

            for n in k+1:nblocks-1
                if n <= m
                    sdatank = get_block(mat, n, k, nblocks)
                    sdatamn = get_block(mat, m, n, nblocks)
                    u22(sdatamk, sdatank, sdatamn)
                end
            end
        end

    end
end

function check(mat::Matrix{Float32})
    size_p = size(mat, 1)

    for i in 1:size_p
        for j in 1:size_p
            if j > i
                mat[i, j] = 0.0f0
            end
        end
    end

    test_mat ::Matrix{Float32} = zeros(Float32, size_p, size_p)

    syrk!('L', 'N', 1.0f0, mat, 0.0f0, test_mat)

    for i in 1:size_p
        for j in 1:size_p
            if j <= i
                orig = (1.0f0/(1.0f0+(i-1)+(j-1))) + ((i == j) ? 1.0f0*size_p : 0.0f0)
                err = abs(test_mat[i,j] - orig) / orig
                if err > 0.0001
                    got = test_mat[i,j]
                    expected = orig
                    error("[$i, $j] -> $got != $expected (err $err)")
                end
            end
        end
    end

    println("Verification successful !")
end

function main(size_p :: Int, nblocks :: Int, display = false)
    mat :: Matrix{Float32} = zeros(Float32, size_p, size_p)

    # create a simple definite positive symetric matrix
    # Hilbert matrix h(i,j) = 1/(i+j+1)

    for i in 1:size_p
        for j in 1:size_p
            mat[i, j] = 1.0f0 / (1.0f0+(i-1)+(j-1)) + ((i == j) ? 1.0f0*size_p : 0.0f0)
        end
    end

    if display
        display(mat)
    end

    t_start = time_ns()

    cholesky(mat, size_p, nblocks)

    t_end = time_ns()

    flop = (1.0*size_p*size_p*size_p)/3.0
    println("# size\tms\tGFlops")
    time_ms = (t_end-t_start) / 1e6
    gflops = flop/(time_ms*1000)/1000
    println("# $size_p\t$time_ms\t$gflops")

    if display
        display(mat)
    end

    check(mat)
end

main(1024*20, 8)

