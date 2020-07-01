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
@codelet function codeletA() :: Nothing
    # print("[Task A] Value = ", val[]);
    # do nothing
end

@target STARPU_CPU
@codelet function codeletB(val ::Ref{Int32}) :: Nothing
    # println("[Task B] Value = ", val[]);
    val[] = val[] *2
end

function callbackB(task)
    sleep(1)
    starpu_task_end_dep_release(task)
end

@target STARPU_CPU
@codelet function codeletC(val ::Ref{Int32}) :: Nothing
    # println("[Task C] Value = ", val[]);
    val[] = val[] *2
end

function callbackC(task)
    starpu_task_end_dep_release(task)
end


function main()
    value = Ref(Int32(12))

    @starpu_block let
        perfmodel = starpu_perfmodel(
            perf_type = starpu_perfmodel_type(STARPU_HISTORY_BASED),
            symbol = "history_perf"
        )

        clA = starpu_codelet(
            cpu_func = "codeletA",
            perfmodel = perfmodel
        )
        clB = starpu_codelet(
            cpu_func = "codeletB",
            modes = [STARPU_RW],
            perfmodel = perfmodel
        )
        clC = starpu_codelet(
            cpu_func = "codeletC",
            modes = [STARPU_RW],
            perfmodel = perfmodel
        )

        handle = starpu_data_register(value)

	starpu_data_set_sequential_consistency_flag(handle, 0)

        taskA = starpu_task(cl = clA, detach=0)
        taskB = starpu_task(cl = clB, handles = [handle], callback=callbackB, callback_arg=taskA)
	taskC = starpu_task(cl = clC, handles = [handle], callback=callbackC, callback_arg=taskA)

	starpu_task_end_dep_add(taskA, 2)
        starpu_task_declare_deps(taskC, taskB)

        starpu_task_submit(taskA)
        starpu_task_submit(taskB)
        starpu_task_submit(taskC)
        starpu_task_wait(taskA)

        starpu_data_acquire_on_node(handle, STARPU_MAIN_RAM, STARPU_R);
	# Waiting for taskA should have also waited for taskB and taskC
        if value[] != 48
            error("Incorrect value $(value[]) (expected 48)")
        end
	starpu_data_release_on_node(handle, STARPU_MAIN_RAM);
    end


    println("Value = ", value[])
end

# Disable garbage collector because of random segfault/hang when using mutex.
# This issue should be solved with Julia release 1.5.
GC.enable(false)
starpu_init()
main()
starpu_shutdown()
GC.enable(true)
