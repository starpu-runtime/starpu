# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#
using StarPU
using LinearAlgebra.BLAS

# Standard kernels for the Cholesky factorization
# U22 is the gemm update
# U21 is the trsm update
# U11 is the cholesky factorization

@target STARPU_CPU+STARPU_CUDA
@codelet function u11(sub11 :: Matrix{Float32}) :: Nothing
    nx :: Int32 = width(sub11)
    ld :: Int32 = ld(sub11)

    for z in 0:nx-1
        lambda11 :: Float32 = sqrt(sub11[z+1,z+1])
        sub11[z+1,z+1] = lambda11

        alpha ::Float32 = 1.0f0 / lambda11
        X :: Vector{Float32} = view(sub11, z+2:z+2+(nx-z-2), z+1)
        STARPU_SSCAL(nx-z-1, alpha, X, 1)

        alpha = -1.0f0
        A :: Matrix{Float32} = view(sub11, z+2:z+2+(nx-z-2), z+2:z+2+(nx-z-2))
	STARPU_SSYR("L", nx-z-1, alpha, X, 1, A, ld)
    end
    return
end

@target STARPU_CPU+STARPU_CUDA
@codelet function u21(sub11 :: Matrix{Float32},
                      sub21 :: Matrix{Float32}) :: Nothing
    ld11 :: Int32 = ld(sub11)
    ld21 :: Int32 = ld(sub21)
    nx21 :: Int32 = width(sub21)
    ny21 :: Int32 = height(sub21)
    alpha :: Float32 = 1.0f0
    STARPU_STRSM("R", "L", "T", "N", nx21, ny21, alpha, sub11, ld11, sub21, ld21)
    return
end

@target STARPU_CPU+STARPU_CUDA
@codelet function u22(left   :: Matrix{Float32},
                      right  :: Matrix{Float32},
                      center :: Matrix{Float32}) :: Nothing
    dx :: Int32 = width(center)
    dy :: Int32 = height(center)
    dz :: Int32 = width(left)
    ld21 :: Int32 = ld(left)
    ld12 :: Int32 = ld(center)
    ld22 :: Int32 = ld(right)
    alpha :: Float32 = -1.0f0
    beta :: Float32 = 1.0f0
    STARPU_SGEMM("N", "T", dy, dx, dz, alpha, left, ld21, right, ld12, beta, center, ld22)
    return
end

function cholesky(mat :: Matrix{Float32}, size, nblocks)
    perfmodel = starpu_perfmodel(
        perf_type = starpu_perfmodel_type(STARPU_HISTORY_BASED),
        symbol = "history_perf"
    )
    cl_11 = starpu_codelet(
        cpu_func = CPU_CODELETS["u11"],
        # This kernel cannot be translated to CUDA yet.
        # cuda_func = CUDA_CODELETS["u11"],
        modes = [STARPU_RW],
        color = 0xffff00,
        perfmodel = perfmodel
    )
    cl_21 = starpu_codelet(
        cpu_func = CPU_CODELETS["u21"],
        # cuda_func = CUDA_CODELETS["u21"],
        modes = [STARPU_R, STARPU_RW],
        color = 0x8080ff,
        perfmodel = perfmodel
    )
    cl_22 = starpu_codelet(
        cpu_func = CPU_CODELETS["u22"],
        # cuda_func = CUDA_CODELETS["u22"],
        modes = [STARPU_R, STARPU_R, STARPU_RW],
        color = 0x00ff00,
        perfmodel = perfmodel
    )

    horiz = starpu_data_filter(STARPU_MATRIX_FILTER_BLOCK, nblocks)
    vert = starpu_data_filter(STARPU_MATRIX_FILTER_VERTICAL_BLOCK, nblocks)

    @starpu_block let
        h_mat = starpu_data_register(mat)
        starpu_data_map_filters(h_mat, horiz, vert)

        for k in 1:nblocks

            starpu_iteration_push(k)

            task = starpu_task(cl = cl_11, handles = [h_mat[k, k]])
            starpu_task_submit(task)

            for m in k+1:nblocks
                task = starpu_task(cl = cl_21, handles = [h_mat[k, k], h_mat[m, k]])
                starpu_task_submit(task)
            end

            for m in k+1:nblocks
                for n in k+1:nblocks
                    if n <= m
                        task = starpu_task(cl = cl_22, handles = [h_mat[m, k], h_mat[n, k], h_mat[m, n]])
                        starpu_task_submit(task)
                    end
                end
            end

            starpu_iteration_pop()
        end

        starpu_task_wait_for_all()
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

function main(size_p :: Int, nblocks :: Int, verbose = false)
    starpu_init()

    mat :: Matrix{Float32} = zeros(Float32, size_p, size_p)

    # create a simple definite positive symetric matrix
    # Hilbert matrix h(i,j) = 1/(i+j+1)

    for i in 1:size_p
        for j in 1:size_p
            mat[i, j] = 1.0f0 / (1.0f0+(i-1)+(j-1)) + ((i == j) ? 1.0f0*size_p : 0.0f0)
        end
    end

    if verbose
        display(mat)
    end

    starpu_memory_pin(mat)

    t_start = time_ns()

    cholesky(mat, size_p, nblocks)

    t_end = time_ns()

    starpu_memory_unpin(mat)

    flop = (1.0*size_p*size_p*size_p)/3.0
    println("# size\tms\tGFlops")
    time_ms = (t_end-t_start) / 1e6
    gflops = flop/(time_ms*1000)/1000
    println("# $size_p\t$time_ms\t$gflops")

    if verbose
        display(mat)
    end

    check(mat)

    starpu_shutdown()
end

main(1024, 8)
