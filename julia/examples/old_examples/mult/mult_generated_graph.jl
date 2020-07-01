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

if length(ARGS) != 6
    println("Usage : julia prog.jl start_dim step_dim stop_dim nb_tests nslicesx nslicesy")
    quit()
end


include("../../src/Wrapper/Julia/starpu_include.jl")
using StarPU




@debugprint "starpu_init"
starpu_init(extern_task_path = "../build/generated_tasks.so")

perfmodel = StarpuPerfmodel(
    perf_type = STARPU_HISTORY_BASED,
    symbol = "history_perf"
)

cl = StarpuCodelet(
    cpu_func = "matrix_mult",
    gpu_func = "CUDA_matrix_mult",
    modes = [STARPU_R, STARPU_R, STARPU_W],
    perfmodel = perfmodel
)

include("mult_def.jl")

display_times(map( (x -> parse(Int64,x)) , ARGS)..., "../mult_generated.dat")

@debugprint "starpu_shutdown"
starpu_shutdown()
