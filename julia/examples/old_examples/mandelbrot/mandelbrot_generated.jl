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


if length(ARGS) != 7
    println("Usage : julia prog.jl cr ci start_dim step_dim stop_dim nslices nbr_tests")
    quit()
end

if (parse(Int64,ARGS[3]) % parse(Int64,ARGS[6]) != 0)
    println("The number of slices should divide all the dimensions.")
    quit()
end

include("../../src/Wrapper/Julia/starpu_include.jl")
using StarPU

@debugprint "starpu_init"
starpu_init(extern_task_path = "../build/generated_tasks_mandelbrot.so")

perfmodel = StarpuPerfmodel(
    perf_type = STARPU_HISTORY_BASED,
    symbol = "history_perf"
)

cl = StarpuCodelet(
    cpu_func = "mandelbrot",
    gpu_func = "CUDA_mandelbrot",
    modes = [STARPU_W, STARPU_R, STARPU_R],
    perfmodel = perfmodel
)

clcpu = StarpuCodelet(
    cpu_func = "mandelbrot",
    modes = [STARPU_W, STARPU_R, STARPU_R],
    perfmodel = perfmodel
)

clgpu = StarpuCodelet(
    gpu_func = "CUDA_mandelbrot",
    modes = [STARPU_W, STARPU_R, STARPU_R],
    perfmodel = perfmodel
)

include("mandelbrot_def.jl")

display_time(parse(Float64,ARGS[1]), parse(Float64,ARGS[2]), map((x -> parse(Int64, x)), ARGS[3:7])...)

@debugprint "starpu_shutdown"
starpu_shutdown()