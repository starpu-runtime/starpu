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

        cl1 = StarpuCodelet(
            cpu_func = CPU_CODELETS["task_insert_color"],
            modes = [STARPU_RW]
        )

        cl2 = StarpuCodelet(
            cpu_func = CPU_CODELETS["task_insert_color"],
            modes = [STARPU_RW],
            color = 0x0000FF
        )

	@starpu_sync_tasks begin

            # In the trace file, the following task should be green (executed on CPU)
            starpu_task_submit(StarpuTask(cl = cl1, handles = [hVal]))

            # In the trace file, the following task will be blue as specified by the field color of cl2
            starpu_task_submit(StarpuTask(cl = cl2, handles = [hVal]))

            # In the trace file, the following tasks will be red as specified in @starpu_async_cl
            @starpu_async_cl task_insert_color(hVal) [STARPU_RW] [] 0xFF0000

	end
    end
end


foo = Ref(convert(Int32, 42))

task_insert_color_with_starpu(foo)

starpu_shutdown()
