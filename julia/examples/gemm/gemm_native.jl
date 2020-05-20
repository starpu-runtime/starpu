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
using LinearAlgebra.BLAS

function gemm_without_starpu(A :: Matrix{Float32}, B :: Matrix{Float32}, C :: Matrix{Float32}, alpha :: Float32, beta :: Float32)
    tmin = 0
    for i in (1 : 10 )
        t=time_ns()
        gemm!('N', 'N', alpha, A, B, beta, C)
        t=time_ns() - t
        if (tmin==0 || tmin>t)
            tmin=t
        end
    end
    return tmin
end


function compute_times(io,start_dim, step_dim, stop_dim)
    for dim in (start_dim : step_dim : stop_dim)
        A = Array(rand(Cfloat, dim, dim))
        B = Array(rand(Cfloat, dim, dim))
        C = zeros(Float32, dim, dim)
        alpha = 4.0f0
        beta = 2.0f0
        mt =  gemm_without_starpu(A, B, C, alpha, beta)
        gflop = 2 * dim * dim * dim * 1.e-9
        gflops = gflop / (mt * 1.e-9)
        size=dim*dim*dim*4*3/1024/1024
        println(io,"$dim $gflops")
        println("$dim $gflops")
    end
end

if size(ARGS, 1) < 1
    filename="x.dat"
else
    filename=ARGS[1]
end
io=open(filename,"w")
compute_times(io,64,512,4096)
close(io)

