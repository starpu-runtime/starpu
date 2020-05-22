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
        starpu_data_map_filters(h_mat, horiz, vert)

        for k in 1:nblocks

            starpu_iteration_push(k)

            starpu_task_insert(cl = cl_11, handles = [h_mat[k, k]], tag_only = tag11(k))

            for m in k+1:nblocks
                starpu_task_insert(cl = cl_21, handles = [h_mat[k, k], h_mat[m, k]], tag_only = tag21(m, k))
            end
            starpu_data_wont_use(h_mat[k, k])

            for m in k+1:nblocks
                for n in k+1:nblocks
                    if n <= m
                        starpu_task_insert(cl = cl_22, handles = [h_mat[m, k], h_mat[n, k], h_mat[m, n]], tag_only= tag22(k, m, n))
                    end
                end
                starpu_data_wont_use(h_mat[m, k])
            end

            starpu_iteration_pop()
        end

        starpu_task_wait_for_all()
    end
end

starpu_init()
starpu_cublas_init()

main(1024, 8, verify = true)
main(15360, 16)

starpu_shutdown()
