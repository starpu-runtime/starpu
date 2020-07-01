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

include("cholesky_common.jl")

function cholesky(mat :: Matrix{Float32}, size, nblocks)
    include("cholesky_codelets.jl")

    horiz = starpu_data_filter(STARPU_MATRIX_FILTER_BLOCK, nblocks)
    vert = starpu_data_filter(STARPU_MATRIX_FILTER_VERTICAL_BLOCK, nblocks)

    @starpu_block let
        h_mat = starpu_data_register(mat)
        starpu_data_set_sequential_consistency_flag(h_mat, 0)
        starpu_data_map_filters(h_mat, horiz, vert)

        entry_task = starpu_task(cl = cl_11,
                                 handles = [h_mat[1, 1]],
                                 tag = tag11(1))

        for k in 1:nblocks

            starpu_iteration_push(k)

            if k > 1
                # enforce dependencies...
                starpu_tag_declare_deps(tag11(k), tag22(k-1, k, k))
                starpu_task_insert(cl = cl_11,
                                   handles = [h_mat[k, k]],
                                   tag = tag11(k))
            end

            for m in k+1:nblocks
                # enforce dependencies...
                if k > 1
                    starpu_tag_declare_deps(tag21(k, m), tag11(k), tag22(k-1, m, k))
                else
                    starpu_tag_declare_deps(tag21(k, m), tag11(k))
                end

                starpu_task_insert(cl = cl_21, handles = [h_mat[k, k], h_mat[m, k]], tag = tag21(k, m))

                for n in k+1:nblocks
                    if n <= m
                        # enforce dependencies...
                        if k > 1
                            starpu_tag_declare_deps(tag22(k, m, n), tag22(k-1, m, n), tag21(k, n), tag21(k, m))
                        else
                            starpu_tag_declare_deps(tag22(k, m, n), tag21(k, n), tag21(k, m))
                        end

                        starpu_task_insert(cl = cl_22, handles = [h_mat[m, k], h_mat[n, k], h_mat[m, n]], tag = tag22(k, m, n))
                    end
                end
            end

            starpu_iteration_pop()
        end

        starpu_task_submit(entry_task)
        starpu_tag_wait(tag11(nblocks))
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
