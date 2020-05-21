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

@target STARPU_CPU
@codelet function task_insert_color(val ::Ref{Int32}) :: Nothing
    val[] = val[] * 2

    return
end

starpu_init()

function task_insert_color_with_starpu(val ::Ref{Int32})
    @starpu_block let
	hVal = starpu_data_register(val)

        perfmodel = starpu_perfmodel(
            perf_type = starpu_perfmodel_type(STARPU_HISTORY_BASED),
            symbol = "history_perf"
        )

        cl1 = starpu_codelet(
            cpu_func = "task_insert_color",
            modes = [STARPU_RW],
            perfmodel = perfmodel
        )

        cl2 = starpu_codelet(
            cpu_func = "task_insert_color",
            modes = [STARPU_RW],
            perfmodel = perfmodel,
            color = 0x0000FF
        )

	@starpu_sync_tasks begin

            # In the trace file, the following task should be green (executed on CPU)
            starpu_task_submit(starpu_task(cl = cl1, handles = [hVal]))

            # In the trace file, the following task will be blue as specified by the field color of cl2
            starpu_task_submit(starpu_task(cl = cl2, handles = [hVal]))

            # In the trace file, the following tasks will be red as specified in @starpu_async_cl
            @starpu_async_cl task_insert_color(hVal) [STARPU_RW] () 0xFF0000

	end
    end
end


foo = Ref(convert(Int32, 42))

task_insert_color_with_starpu(foo)

starpu_shutdown()
