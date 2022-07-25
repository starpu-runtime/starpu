# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020, 2022       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

include("cholesky_common.jl")

function cholesky(mat :: Matrix{Float32}, size, nblocks)
    include("cholesky_codelets.jl")

    horiz = starpu_data_filter(STARPU_MATRIX_FILTER_BLOCK, nblocks)
    vert = starpu_data_filter(STARPU_MATRIX_FILTER_VERTICAL_BLOCK, nblocks)

    @starpu_block let
        h_mat = starpu_data_register(mat)
        starpu_data_set_sequential_consistency_flag(h_mat, 0)
        starpu_data_map_filters(h_mat, horiz, vert)

        entry_task = starpu_task(cl = cl_potrf,
                                 handles = [h_mat[1, 1]],
                                 tag = tag_potrf(1))

        for k in 1:nblocks

            starpu_iteration_push(k)

            if k > 1
                # enforce dependencies...
                starpu_tag_declare_deps(tag_potrf(k), tag_gemm(k-1, k, k))
                starpu_task_insert(cl = cl_potrf,
                                   handles = [h_mat[k, k]],
                                   tag = tag_potrf(k))
            end

            for m in k+1:nblocks
                # enforce dependencies...
                if k > 1
                    starpu_tag_declare_deps(tag_trsm(k, m), tag_potrf(k), tag_gemm(k-1, m, k))
                else
                    starpu_tag_declare_deps(tag_trsm(k, m), tag_potrf(k))
                end

                starpu_task_insert(cl = cl_trsm, handles = [h_mat[k, k], h_mat[m, k]], tag = tag_trsm(k, m))

                for n in k+1:nblocks
                    if n <= m
                        # enforce dependencies...
                        if k > 1
                            starpu_tag_declare_deps(tag_gemm(k, m, n), tag_gemm(k-1, m, n), tag_trsm(k, n), tag_trsm(k, m))
                        else
                            starpu_tag_declare_deps(tag_gemm(k, m, n), tag_trsm(k, n), tag_trsm(k, m))
                        end

                        starpu_task_insert(cl = cl_gemm, handles = [h_mat[m, k], h_mat[n, k], h_mat[m, n]], tag = tag_gemm(k, m, n))
                    end
                end
            end

            starpu_iteration_pop()
        end

        starpu_task_submit(entry_task)
        starpu_tag_wait(tag_potrf(nblocks))
    end
end

starpu_init()
starpu_cublas_init()

println("# size\tms\tGFlops")

if length(ARGS) > 0 && ARGS[1] == "-quickcheck"
    main(1024, 8, verify = true)
else
    for size in 1024:1024:15360
        main(size, 16)
    end
end

starpu_shutdown()
