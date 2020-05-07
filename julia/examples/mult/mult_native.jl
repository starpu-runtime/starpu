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
import Libdl
using StarPU
using LinearAlgebra

function multiply_without_starpu(A :: Matrix{Float32}, B :: Matrix{Float32}, C :: Matrix{Float32}, nslicesx, nslicesy, stride)
    tmin = 0
    for i in (1 : 10 )
        t=time_ns()
        C = A * B;
        t=time_ns() - t
        if (tmin==0 || tmin>t)
            tmin=t
        end
    end
    return tmin
end


function compute_times(io,start_dim, step_dim, stop_dim, nslicesx, nslicesy, stride)
    for dim in (start_dim : step_dim : stop_dim)
        A = Array(rand(Cfloat, dim, dim))
        B = Array(rand(Cfloat, dim, dim))
        C = zeros(Float32, dim, dim)
        mt =  multiply_without_starpu(A, B, C, nslicesx, nslicesy, stride)
        flops = (2*dim-1)*dim*dim/mt
        size=dim*dim*4*3/1024/1024
        println(io,"$size $flops")
        println("$size $flops")
    end
end

if size(ARGS, 1) < 2
    stride=4
    filename="x.dat"
else
    stride=parse(Int, ARGS[1])
    filename=ARGS[2]
end
io=open(filename,"w")
compute_times(io,16*stride,4*stride,128*stride,2,2,stride)
close(io)

