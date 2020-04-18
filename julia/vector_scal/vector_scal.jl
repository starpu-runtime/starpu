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


@debugprint "starpu_init"
starpu_init()

function vector_scal_with_starpu(v :: Vector{Float32}, m :: Int32, k :: Float32, l :: Float32)
    tmin=0

    @starpu_block let
        hV = starpu_data_register(v)
        tmin=0
        perfmodel = StarpuPerfmodel(
            perf_type = STARPU_HISTORY_BASED,
            symbol = "history_perf"
        )
        cl = StarpuCodelet(
            cpu_func = CPU_CODELETS["vector_scal"],
            # cuda_func = CUDA_CODELETS["vector_scal"],
            #opencl_func="ocl_matrix_mult",
            modes = [STARPU_RW],
            perfmodel = perfmodel
        )

        for i in (1 : 1)
            t=time_ns()
            @starpu_sync_tasks begin
                handles = [hV]
                task = StarpuTask(cl = cl, handles = handles, cl_arg=[m, k, l])
                starpu_task_submit(task)
            end
            # @starpu_sync_tasks for task in (1:1)
            #     @starpu_async_cl vector_scal(hV, STARPU_RW, [m, k, l])
            # end
            t=time_ns()-t
            if (tmin==0 || tmin>t)
                tmin=t
            end
        end
    end
    return tmin
end

function compute_times(io,start_dim, step_dim, stop_dim)
    for size in (start_dim : step_dim : stop_dim)
        V = Array(rand(Cfloat, size))
        m :: Int32 = 10
        k :: Float32 = 2.
        l :: Float32 = 3.
        println("INPUT ", V[1:10])
        mt =  vector_scal_with_starpu(V, m, k, l)
        println("OUTPUT ", V[1:10])
        println(io,"$size $mt")
        println("$size $mt")
    end
end


io=open(ARGS[1],"w")
compute_times(io,1024,1024,4096)
close(io)
@debugprint "starpu_shutdown"
starpu_shutdown()

