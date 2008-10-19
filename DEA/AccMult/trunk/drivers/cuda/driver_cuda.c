#include "driver_cuda.h"
#include <core/policies/sched_policy.h>

/* the number of CUDA devices */
int ncudagpus;

//static CUdevice cuDevice;
static CUcontext cuContext[MAXCUDADEVS];
CUresult status;

//CUdeviceptr debugptr;

extern char *execpath;

void init_cuda_module(struct cuda_module_s *module, char *path)
{
	unsigned i;
	for (i = 0; i < MAXCUDADEVS; i++)
	{
		module->is_loaded[i] = 0;
	}

	module->module_path = path;
}

void load_cuda_module(int devid, struct cuda_module_s *module)
{
	CUresult res;
	if (!module->is_loaded[devid])
	{
		res = cuModuleLoad(&module->module, module->module_path);
		if (res) {
			CUDA_REPORT_ERROR(res);
		}
	
		module->is_loaded[devid] = 1;
	}
}

void init_cuda_function(struct cuda_function_s *func, 
			struct cuda_module_s *module,
			char *symbol)
{
	unsigned i;
	for (i = 0; i < MAXCUDADEVS; i++)
	{
		func->is_loaded[i] = 0;
	}

	func->symbol = symbol;
	func->module = module;
}

void set_function_args(cuda_codelet_t *args, 
			buffer_descr *descr,
			data_interface_t *interface, 
			unsigned nbuffers)
{
	CUresult res;

	unsigned offset = 0;

//	res = cuParamSetv(args->func->function, offset, 
//		&debugptr, sizeof(uint64_t *));
//	if (res != CUDA_SUCCESS) {
//		CUDA_REPORT_ERROR(res);
//	}
//	offset += sizeof(uint64_t *);
//


	unsigned buf;
	for (buf = 0; buf < nbuffers; buf++)
	{
		size_t size;
		/* this buffer is filled with the stack to be given to the GPU 
		 * the size of the buffer may be changed if needed */
		uint8_t argbuffer[128];

		data_state *state = descr[buf].state;
	
		/* dump the stack into the buffer */
		ASSERT(state);
		ASSERT(state->ops);
		ASSERT(state->ops->dump_data_interface);

		size = state->ops->dump_data_interface(&interface[buf], argbuffer);

		res = cuParamSetv(args->func->function, offset, (void *)argbuffer, size);
		if (res != CUDA_SUCCESS) {
			CUDA_REPORT_ERROR(res);
		}
		offset += size;
	}

	if (args->stack_size) {
		res = cuParamSetv(args->func->function, offset, 
			args->stack, args->stack_size);
		if (res != CUDA_SUCCESS) {
			CUDA_REPORT_ERROR(res);
		}
		offset += args->stack_size;
	}

	res = cuParamSetSize(args->func->function, offset);
	if (res != CUDA_SUCCESS) {
		CUDA_REPORT_ERROR(res);
	}

	unsigned shmsize = args->shmemsize;
	res = cuFuncSetSharedSize(args->func->function, shmsize);
	if (res != CUDA_SUCCESS) {
		CUDA_REPORT_ERROR(res);
	}
}

void load_cuda_function(int devid, struct cuda_function_s *function)
{
	CUresult res;

	/* load the module on the device if it is not already the case */
	load_cuda_module(devid, function->module);

	/* load the function on the device if it is not present yet */
	res = cuModuleGetFunction( &function->function, 
			function->module->module, function->symbol );
	if (res) {
		CUDA_REPORT_ERROR(res);
	}

}

void init_context(int devid)
{
	status = cuCtxCreate( &cuContext[devid], 0, 0);
	if (status) {
		CUDA_REPORT_ERROR(status);
	}

	status = cuCtxAttach(&cuContext[devid], 0);
	if (status) {
		CUDA_REPORT_ERROR(status);
	}

	cublasInit();
}

unsigned get_cuda_device_count(void)
{
	int cnt;
	cuDeviceGetCount(&cnt);
	return (unsigned)cnt;
}

void init_cuda(void)
{
	CUresult status;

	status = cuInit(0);
	if (status) {
		CUDA_REPORT_ERROR(status);
	}

	cuDeviceGetCount(&ncudagpus);
	assert(ncudagpus <= MAXCUDADEVS);

	int dev;
	for (dev = 0; dev < ncudagpus; dev++)
	{
		// TODO change this to the driver API
		// cudaGetDeviceProperties(&cudadevprops[dev], dev);
	}
}

int execute_job_on_cuda(job_t j, int devid, unsigned use_cublas)
{
	int ret;
	uint32_t mask = (1<<0);

	CUresult status;
	tick_t codelet_start, codelet_end;
	tick_t codelet_start_comm, codelet_end_comm;
	
	switch (j->type) {
		case CODELET:
			ASSERT(j);
			ASSERT(j->cl);

			GET_TICK(codelet_start_comm);

			ret = fetch_codelet_input(j->buffers, j->interface, j->nbuffers, mask);
			if (ret != 0) {
				/* there was not enough memory, so the input of the codelet cannot be fetched ... put the codelet back, and try it later */
				return TRYAGAIN;
			}

			TRACE_START_CODELET_BODY(j);
			if (use_cublas) {
				cl_func func = j->cl->cublas_func;
				ASSERT(func);
				GET_TICK(codelet_start);
				func(j->interface, j->cl->cl_arg);
				cuCtxSynchronize();
				GET_TICK(codelet_end);
			} else {
				/* load the module and the function */
				cuda_codelet_t *args; 
				args = j->cl->cuda_func;

				load_cuda_function(devid, args->func);

				status = cuFuncSetBlockShape(args->func->function,
							args->blockx, 
							args->blocky, 1);
				if (status) {
					CUDA_REPORT_ERROR(status);
				}

				/* set up the function args */
				set_function_args(args, j->buffers, j->interface, j->nbuffers);

				/* set up the grids */
#ifdef MODEL_DEBUG
				status = cuCtxSynchronize();
#endif
				GET_TICK(codelet_start);
				status = cuLaunchGrid(args->func->function, 
						args->gridx, args->gridy);
				if (status) {
					CUDA_REPORT_ERROR(status);
				}


				/* launch the function */
				status = cuCtxSynchronize();
				if (status) {
					CUDA_REPORT_ERROR(status);
				}
				GET_TICK(codelet_end);

			}
			TRACE_END_CODELET_BODY(j);	
			push_codelet_output(j->buffers, j->nbuffers, mask);

			GET_TICK(codelet_end_comm);

#ifdef MODEL_DEBUG
			double measured = timing_delay(&codelet_start, &codelet_end);
			double measured_comm = timing_delay(&codelet_start_comm, &codelet_end_comm);

			if (j->predicted != 0.0) 
			{
				fprintf(stderr, "CUDA : model was %e, got %e (with comms, %e), factor (%2.2f \%%)\n", 
					j->predicted, measured, measured_comm, 100*(measured/j->predicted - 1.0f));
			}
#endif

			break;
		case ABORT:
			printf("CUDA abort\n");
			cublasShutdown();
			thread_exit(NULL);
			break;
		default:
			break;
	}

	return OK;
}

void *cuda_worker(void *arg)
{
	struct cuda_worker_arg_t* args = (struct cuda_worker_arg_t*)arg;

	int devid = args->deviceid;

#ifdef USE_FXT
	fxt_register_thread(((struct cuda_worker_arg_t *)arg)->bindid);
#endif
	TRACE_NEW_WORKER(FUT_CUDA_KEY);

#ifndef DONTBIND
        /* fix the thread on the correct cpu */
        cpu_set_t aff_mask;
        CPU_ZERO(&aff_mask);
        CPU_SET(args->bindid, &aff_mask);
        sched_setaffinity(0, sizeof(aff_mask), &aff_mask);
#endif

	set_local_memory_node_key(&(((cuda_worker_arg *)arg)->memory_node));

	set_local_queue(args->jobq);

	init_context(devid);
	fprintf(stderr, "cuda thread is ready to run on CPU %d !\n", args->bindid);

//	uint64_t foo = 1664;
//	cuMemAlloc(&debugptr, sizeof(uint64_t));
//	cuMemcpyHtoD(debugptr, &foo, sizeof(uint64_t));
	

	/* tell the main thread that this one is ready */
	args->ready_flag = 1;

	struct job_s * j;
	int res;
	
	do {
		//int debugfoo;
		j = pop_task();
		//printf("cuda driver picked %p\n", j);
		if (j == NULL) continue;

		/* can CUDA do that task ? */
		if (!CUDA_MAY_PERFORM(j) && !CUBLAS_MAY_PERFORM(j))
		{
			/* this is neither a cuda or a cublas task */
			push_task(j);
			continue;
		}

//		cuMemcpyDtoH(&debugfoo, debugptr, sizeof(uint64_t));
//		printf("BEFORE TASK, debug ptr = %p\n", debugfoo);


		unsigned use_cublas = CUBLAS_MAY_PERFORM(j) ? 1:0;
		res = execute_job_on_cuda(j, devid, use_cublas);

		if (res != OK) {
			switch (res) {
				case OK:
				case FATAL:
					assert(0);
				case TRYAGAIN:
					printf("ouch, put the codelet %p back ... \n", j);
					push_task(j);
					ASSERT(0);
					continue;
				default:
					assert(0);
			}
		}

		if (j->cb)
			j->cb(j->argcb);

                /* in case there are dependencies, wake up the proper tasks */
                notify_dependencies(j);

//		cuMemcpyDtoH(&debugfoo, debugptr, sizeof(uint64_t));
//		printf("AFTER TASK, debug ptr = %p\n", debugfoo);


		job_delete(j);
		//printf("cuda terminated %p\n", j);

	} while(1);

	return NULL;

//error:
//	CUDA_REPORT_ERROR(status);
//	assert(0);

}
