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

function callbackA(arg)
    clB = arg[1]
    handle = arg[2]
    tagHoldC = arg[3]

    taskB = starpu_task(cl = clB, handles = [handle],
                        callback = starpu_tag_notify_from_apps,
                        callback_arg = tagHoldC,
                        sequential_consistency=false)

    starpu_task_submit(taskB)
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


# Submit taskA and hold it
# Submit taskC and hold it
# Release taskA
# Execute taskA       --> callback: submit taskB
# Execute taskB       --> callback: release taskC
#
# All three tasks use the same data in RW, taskB is submitted after
# taskC, so taskB should normally only execute after taskC but as the
# sequential consistency for (taskB, data) is unset, taskB can
# execute straightaway
function main()
    value = Ref(Int32(12))

    @starpu_block let
    tagHoldA :: starpu_tag_t = 32
    tagHoldC :: starpu_tag_t = 84
    tagA :: starpu_tag_t = 421
    tagC :: starpu_tag_t = 842

    starpu_tag_declare_deps(tagA, tagHoldA)
    starpu_tag_declare_deps(tagC, tagHoldC)

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

        handle = starpu_data_register(value)

        taskA = starpu_task(cl = clA, handles = [handle], tag = tagA,
                            callback = callbackA,
                            callback_arg=(clB, handle, tagHoldC))
        starpu_task_submit(taskA)

        taskC = starpu_task(cl = clC, handles = [handle], tag = tagC)
        starpu_task_submit(taskC)

        # Release taskA (we want to make sure it will execute after taskC has been submitted)
        starpu_tag_notify_from_apps(tagHoldA)

        starpu_task_wait_for_all()
    end

    if value[] != 50
        error("Incorrect value $(value[]) (expected 50)")
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
