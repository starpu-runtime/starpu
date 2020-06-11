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
using Printf
const EPSILON = 1e-6

function check(alpha, X, Y)
    for i in 1:length(X)
        expected_value = alpha * X[i] + 4.0
        if abs(Y[i] - expected_value) > expected_value * EPSILON
            error("at ", i, ", ", alpha, "*", X[i], "+4.0=", Y[i], ", expected ", expected_value)
        end
    end
end

@target STARPU_CPU+STARPU_CUDA
@codelet function axpy(X :: Vector{Float32}, Y :: Vector{Float32}, alpha ::Float32) :: Nothing
    STARPU_SAXPY(length(X), alpha, X, 1, Y, 1)
    return
end

function axpy(N, NBLOCKS, alpha, display = true)
    X = Array(fill(1.0f0, N))
    Y = Array(fill(4.0f0, N))

    starpu_memory_pin(X)
    starpu_memory_pin(Y)

    block_filter = starpu_data_filter(STARPU_VECTOR_FILTER_BLOCK, NBLOCKS)

    if display
        println("BEFORE x[0] = ", X[1])
        println("BEFORE y[0] = ", Y[1])
    end

    t_start = time_ns()

    @starpu_block let
        hX,hY = starpu_data_register(X, Y)

        starpu_data_partition(hX, block_filter)
        starpu_data_partition(hY, block_filter)

        for b in 1:NBLOCKS
            starpu_task_insert(codelet_name = "axpy",
                               handles = [hX[b], hY[b]],
                               cl_arg = (Float32(alpha),),
                               tag = starpu_tag_t(b),
                               modes = [STARPU_R, STARPU_RW])
        end

        starpu_task_wait_for_all()
    end

    t_end = time_ns()

    timing = (t_end-t_start)/1000

    if display
        @printf("timing -> %d us %.2f MB/s\n", timing, 3*N*4/timing)
        println("AFTER y[0] = ", Y[1], " (ALPHA=", alpha, ")")
    end

    check(alpha, X, Y)

    starpu_memory_unpin(X)
    starpu_memory_unpin(Y)
end

function main()
    N = 16 * 1024 * 1024
    NBLOCKS = 8
    alpha = 3.41

    starpu_init()
    starpu_cublas_init()

    # warmup
    axpy(10, 1, alpha, false)

    axpy(N, NBLOCKS, alpha)

    starpu_shutdown()
end

main()
