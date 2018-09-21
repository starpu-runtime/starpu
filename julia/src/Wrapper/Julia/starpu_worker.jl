


@enum(StarpuWorkerArchtype,

	STARPU_CPU_WORKER,
	STARPU_CUDA_WORKER,
	STARPU_OPENCL_WORKER,
	STARPU_MIC_WORKER,
	STARPU_SCC_WORKER,
	STARPU_MPI_MS_WORKER,
	STARPU_ANY_WORKER
)


function starpu_worker_get_count_by_type(arch_type :: StarpuWorkerArchtype)
    @starpucall(starpu_worker_get_count_by_type,
            Cint, (StarpuWorkerArchtype,), arch_type
        )
end


#= TODO : NOT C_NULL but stdout FILE *
function starpu_worker_display_names(arch_type :: StarpuWorkerArchtype)
	@starpucall(starpu_worker_display_names,
            Void, (Ptr{Void}, StarpuWorkerArchtype),
			C_NULL, arch_type
        )
end
=#




@starpu_noparam_function "starpu_worker_get_id" Cint

@starpu_noparam_function "starpu_cpu_worker_get_count" Cuint
@starpu_noparam_function "starpu_cuda_worker_get_count" Cuint
