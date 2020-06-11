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

@target STARPU_CPU+STARPU_CUDA
@codelet function gemm(A :: Matrix{Float32}, B :: Matrix{Float32}, C :: Matrix{Float32}, alpha :: Float32, beta :: Float32) :: Nothing

    M :: Int32 = height(A)
    N :: Int32 = width(B)
    K :: Int32 = width(A)
    lda :: Int32 = ld(A)
    ldb :: Int32 = ld(B)
    ldc :: Int32 = ld(C)
    STARPU_SGEMM("N", "N", M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)

    return
end

function multiply_with_starpu(A :: Matrix{Float32}, B :: Matrix{Float32}, C :: Matrix{Float32}, alpha :: Float32, beta :: Float32, nslicesx, nslicesy)
    scale= 3
    tmin=0
    hA,hB,hC = starpu_data_register(A, B, C)
    tmin=0
    perfmodel = starpu_perfmodel(
        perf_type = starpu_perfmodel_type(STARPU_HISTORY_BASED),
        symbol = "gemm"
    )
    cl = starpu_codelet(
        cpu_func  = "gemm",
        cuda_func = "",
        modes =[STARPU_R,STARPU_R,STARPU_RW], 
        perfmodel = perfmodel,
    )
    task = starpu_task(cl = cl, handles =[hA,hB,hC], cl_arg = (alpha,beta), callback = nothing,
		callback_arg = nothing, tag = nothing, tag_only = nothing,
                       sequential_consistency = true,
                       detach = 1, color = nothing, where = nothing)


    for i in (1 : 10 )
        t=time_ns()
starpu_task_submit(Ref(task.c_task))
        #starpu_task_submit(task)
        starpu_task_wait_for_all()
        t=time_ns()-t
	if (tmin==0 || tmin>t)
           tmin=t
        end
    end
    starpu_data_unregister(hA)
    starpu_data_unregister(hB)
    starpu_data_unregister(hC)
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

function check(expected, A, B, C, alpha, beta)
    for i in 1 : 10
        gemm!('N', 'N', alpha, A, B, beta, expected)
    end

    height,width = size(C)
    for i in 1:height
        for j in 1:width
            got = C[i, j]
            exp = expected[i, j]

            err = abs(exp - got) / exp
            if err > 0.0001
                error("[$i] -> $got != $exp (err $err)")
            end
        end
    end
end

function compute_times(io,start_dim, step_dim, stop_dim, nslicesx, nslicesy)
    for dim in (start_dim : step_dim : stop_dim)
        A = Array(rand(Cfloat, dim, dim))
        B = Array(rand(Cfloat, dim, dim))
        C = zeros(Float32, dim, dim)
        C_ref = copy(C)
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
        #check(C_ref, A, B, C, alpha, beta)
    end
end

if size(ARGS, 1) < 1
    filename="x.dat"
else
    filename=ARGS[1]
end

starpu_init()
starpu_cublas_init()
io=open(filename,"w")
compute_times(io,64,512,4096,1,1)
close(io)

starpu_shutdown()

