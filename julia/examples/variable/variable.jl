import Libdl
using StarPU

@target STARPU_CPU
@codelet function variable(val ::Ref{Float32}) :: Nothing
    val[] = val[] + 1

    return
end

starpu_init()

function variable_with_starpu(val ::Ref{Float32}, niter)
    @starpu_block let
	hVal = starpu_data_register(val)

	@starpu_sync_tasks for task in (1 : niter)
                @starpu_async_cl variable(hVal) [STARPU_RW]
	end
    end
end

function display(niter)
    foo = Ref(0.0f0)

    variable_with_starpu(foo, niter)

    println("variable -> ", foo[])
    if foo[] == niter
        println("result is correct")
    else
        println("result is incorret")
    end
end

display(10)

starpu_shutdown()
