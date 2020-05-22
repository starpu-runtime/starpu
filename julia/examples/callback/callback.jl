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

@target STARPU_CPU
@codelet function variable(val ::Ref{Int32}) :: Nothing
    val[] = val[] + 1

    return
end

function callback(args)
    cl = args[1]
    handles = args[2]

    task = starpu_task(cl = cl, handles=handles)
    starpu_task_submit(task)
end

function variable_with_starpu(val ::Ref{Int32})
    perfmodel = starpu_perfmodel(
        perf_type = starpu_perfmodel_type(STARPU_HISTORY_BASED),
        symbol = "history_perf"
    )

    cl = starpu_codelet(
        cpu_func = "variable",
        modes = [STARPU_RW],
        perfmodel = perfmodel
    )

    @starpu_block let
	hVal = starpu_data_register(val)

        starpu_task_insert(codelet_name = "variable",
                           cl = cl,
                           handles = [hVal],
                           callback = callback,
                           callback_arg = (cl, [hVal]))

        starpu_task_wait_for_all()
    end
end

function display()
    v = Ref(Int32(40))

    variable_with_starpu(v)

    println("variable -> ", v[])
    if v[] == 42
        println("result is correct")
    else
        error("result is incorret")
    end
end

# Disable garbage collector because of random segfault/hang when using mutex.
# This issue should be solved with Julia release 1.5.
GC.enable(false)
starpu_init()
display()
starpu_shutdown()
GC.enable(true)
