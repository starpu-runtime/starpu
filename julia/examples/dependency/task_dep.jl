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
@codelet function codeletA(val ::Ref{Int32}) :: Nothing
    # print("[Task A] Value = ", val[]);
    val[] = val[] * 2
end

@target STARPU_CPU
@codelet function codeletB(val ::Ref{Int32}) :: Nothing
    # println("[Task B] Value = ", val[]);
    val[] = val[] +1
end

@target STARPU_CPU
@codelet function codeletC(val ::Ref{Int32}) :: Nothing
    # println("[Task C] Value = ", val[]);
    val[] = val[] *2
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
            modes = [STARPU_RW],
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

        starpu_data_set_default_sequential_consistency_flag(0)

        handle = starpu_data_register(value)

        taskA = starpu_task(cl = clA, handles = [handle])
        taskB = starpu_task(cl = clB, handles = [handle])
        taskC = starpu_task(cl = clC, handles = [handle])

        starpu_task_declare_deps(taskA, taskB)
        starpu_task_declare_deps(taskC, taskA, taskB)

        starpu_task_submit(taskA)
        starpu_task_submit(taskB)
        starpu_task_submit(taskC)

        starpu_task_wait_for_all()
    end

    if value[] != 52
        error("Incorrect value $(value[]) (expected 52)")
    end

    println("Value = ", value[])
end

starpu_init()
main()
starpu_shutdown()
