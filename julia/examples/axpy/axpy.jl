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

const EPSILON = 1e-6

function check(alpha, X, Y)
    for i in 1:length(X)
        expected_value = alpha * X[i] + 4.0
        if abs(Y[i] - expected_value) > expected_value * EPSILON
            error("at ", i, ", ", alpha, "*", X[i], "+4.0=", Y[i], ", expected ", expected_value)
        end
    end
end

function main()
    N = 16 * 1024 * 1024
    NBLOCKS = 8
    alpha = 3.41

    starpu_init()
    starpu_cublas_init()

    X = Array(fill(1.0f0, N))
    Y = Array(fill(4.0f0, N))

    starpu_memory_pin(X)
    starpu_memory_pin(Y)

    println("BEFORE x[0] = ", X[1])
    println("BEFORE y[0] = ", Y[1])

    block_filter = starpu_data_filter(STARPU_VECTOR_FILTER_BLOCK, NBLOCKS)

    perfmodel = starpu_perfmodel(
        perf_type = starpu_perfmodel_type(STARPU_HISTORY_BASED),
        symbol = "history_perf"
    )

    cl = starpu_codelet(
        cpu_func = STARPU_SAXPY,
        cuda_func = STARPU_SAXPY,
        modes = [STARPU_R, STARPU_RW],
        perfmodel = perfmodel
    )

    @starpu_block let
        hX,hY = starpu_data_register(X, Y)

        starpu_data_partition(hX, block_filter)
        starpu_data_partition(hY, block_filter)

        t_start = time_ns()

        for b in 1:NBLOCKS
            task = starpu_task(cl = cl, handles = [hX[b],hY[b]], cl_arg=(Float32(alpha),),
                               tag=starpu_tag_t(b))
            starpu_task_submit(task)
        end
        starpu_task_wait_for_all()

        t_end = time_ns()
        timing = (t_end - t_start) / 1000

        println("timing -> ", timing, " us ", 3*N*4/timing, "MB/s")

    end

    println("AFTER y[0] = ", Y[1], " (ALPHA=", alpha, ")")

    check(alpha, X, Y)

    starpu_shutdown()
end

main()
