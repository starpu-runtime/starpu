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

@target STARPU_CPU+STARPU_CUDA
@codelet function gemm(A :: Matrix{Float32}, B :: Matrix{Float32}, C :: Matrix{Float32}, alpha :: Float32, beta :: Float32) :: Nothing

    M :: Int32 = height(A)
    N :: Int32 = width(B)
    K :: Int32 = width(A)
    lda :: Int32 = height(A)
    ldb :: Int32 = height(B)
    ldc :: Int32 = height(C)
    STARPU_SGEMM("N", "N", M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)

    return
end

function multiply_with_starpu(A :: Matrix{Float32}, B :: Matrix{Float32}, C :: Matrix{Float32}, alpha :: Float32, beta :: Float32, nslicesx, nslicesy)
    scale= 3
    tmin=0
    vert = starpu_data_filter(STARPU_MATRIX_FILTER_VERTICAL_BLOCK, nslicesx)
    horiz = starpu_data_filter(STARPU_MATRIX_FILTER_BLOCK, nslicesy)
    @starpu_block let
        hA,hB,hC = starpu_data_register(A, B, C)
        starpu_data_partition(hB, vert)
        starpu_data_partition(hA, horiz)
        starpu_data_map_filters(hC, vert, horiz)
        tmin=0
        perfmodel = starpu_perfmodel(
            perf_type = starpu_perfmodel_type(STARPU_HISTORY_BASED),
            symbol = "history_perf"
        )
        cl = starpu_codelet(
            cpu_func = "gemm",
            cuda_func = "gemm",
            modes = [STARPU_R, STARPU_R, STARPU_RW],
            perfmodel = perfmodel
        )

        for i in (1 : 10 )
            t=time_ns()
            @starpu_sync_tasks begin
                for taskx in (1 : nslicesx)
                    for tasky in (1 : nslicesy)
                        handles = [hA[tasky], hB[taskx], hC[taskx, tasky]]
                        task = starpu_task(cl = cl, handles = handles, cl_arg=(alpha, beta))
                        starpu_task_submit(task)
                        #@starpu_async_cl matrix_mult(hA[tasky], hB[taskx], hC[taskx, tasky])
                    end
                end
            end
            t=time_ns()-t
            if (tmin==0 || tmin>t)
                tmin=t
            end
        end
    end
    return tmin
end


function approximately_equals(
    A :: Matrix{Cfloat},
    B :: Matrix{Cfloat},
    eps = 1e-2
)
    (height, width) = size(A)

    for j in (1 : width)
        for i in (1 : height)
            if (abs(A[i,j] - B[i,j]) > eps * max(abs(B[i,j]), abs(A[i,j])))
                println("A[$i,$j] : $(A[i,j]), B[$i,$j] : $(B[i,j])")
                return false
            end
        end
    end

    return true
end

function compute_times(io,start_dim, step_dim, stop_dim, nslicesx, nslicesy)
    for dim in (start_dim : step_dim : stop_dim)
        A = Array(rand(Cfloat, dim, dim))
        B = Array(rand(Cfloat, dim, dim))
        C = zeros(Float32, dim, dim)
        starpu_memory_pin(A)
        starpu_memory_pin(B)
        starpu_memory_pin(C)
        alpha = 4.0f0
        beta = 2.0f0
        mt =  multiply_with_starpu(A, B, C, alpha, beta, nslicesx, nslicesy)
        gflop = 2 * dim * dim * dim * 1.e-9
        gflops = gflop / (mt * 1.e-9)
        size=dim*dim*dim*4*3/1024/1024
        println(io,"$dim $gflops")
        println("$dim $gflops")
        starpu_memory_unpin(A)
        starpu_memory_unpin(B)
        starpu_memory_unpin(C)
    end
end

if size(ARGS, 1) < 1
    filename="x.dat"
else
    filename=ARGS[1]
end

starpu_init()

io=open(filename,"w")
compute_times(io,64,512,4096,2,2)
close(io)

starpu_shutdown()

