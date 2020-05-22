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

@target STARPU_CPU+STARPU_CUDA
@codelet function vector_scal(m::Int32, v :: Vector{Float32}, k :: Float32, l :: Float32) :: Float32

    N :: Int32 = length(v)
    # Naive version
    @parallel for i in (1 : N)
        v[i] = v[i] * m + l + k
    end
end


starpu_init()

function vector_scal_with_starpu(v :: Vector{Float32}, m :: Int32, k :: Float32, l :: Float32)
    tmin=0

    @starpu_block let
        hV = starpu_data_register(v)
        tmin=0

        for i in (1 : 1)
            t=time_ns()
            @starpu_sync_tasks begin
                starpu_task_insert(codelet_name = "vector_scal",
                                   modes = [STARPU_RW],
                                   handles = [hV],
                                   cl_arg=(m, k, l))
            end
            t=time_ns()-t
            if (tmin==0 || tmin>t)
                tmin=t
            end
        end
    end
    return tmin
end

function check(ref, res, m, k, l)
    expected = ref .* m .+ (k+l)

    for i in 1:length(expected)
        got = res[i]
        exp = expected[i]

        err = abs(exp - got) / exp
        if err > 0.0001
            error("[$i] -> $got != $exp (err $err)")
        end
    end
end

function compute_times(io,start_dim, step_dim, stop_dim)
    for size in (start_dim : step_dim : stop_dim)
        V = Array(rand(Cfloat, size))
        V_ref = copy(V)
        starpu_memory_pin(V)

        m :: Int32 = 10
        k :: Float32 = 2.
        l :: Float32 = 3.

        println("INPUT ", V[1:10])

        mt =  vector_scal_with_starpu(V, m, k, l)

        starpu_memory_unpin(V)

        println("OUTPUT ", V[1:10])
        println(io,"$size $mt")
        println("$size $mt")

        check(V_ref, V, m, k, l)
    end
end

if size(ARGS, 1) < 1
    filename="x.dat"
else
    filename=ARGS[1]
end

io=open(filename,"w")
compute_times(io,1024,1024,4096)
close(io)

starpu_shutdown()

