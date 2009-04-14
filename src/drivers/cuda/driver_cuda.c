/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include "driver_cuda.h"
#include <core/policies/sched_policy.h>

/* the number of CUDA devices */
static int ncudagpus;

//static CUdevice cuDevice;
static CUcontext cuContext[MAXCUDADEVS];
CUresult status;

//CUdeviceptr debugptr;

extern char *execpath;

void starpu_init_cuda_module(struct starpu_cuda_module_s *module, char *path)
{
	unsigned i;
	for (i = 0; i < MAXCUDADEVS; i++)
	{
		module->is_loaded[i] = 0;
	}

	module->module_path = path;
}

void starpu_load_cuda_module(int devid, struct starpu_cuda_module_s *module)
{
	CUresult res;
	if (!module->is_loaded[devid])
	{
		res = cuModuleLoad(&module->module, module->module_path);
		if (res) {
			fprintf(stderr, "cuModuleLoad failed to open %s\n",
					module->module_path);
			CUDA_REPORT_ERROR(res);
		}
	
		module->is_loaded[devid] = 1;
	}
}

void starpu_init_cuda_function(struct starpu_cuda_function_s *func, 
			struct starpu_cuda_module_s *module,
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

void set_function_args(starpu_cuda_codelet_t *args, 
			starpu_buffer_descr *descr,
			starpu_data_interface_t *interface, 
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
		STARPU_ASSERT(state);
		STARPU_ASSERT(state->ops);
		STARPU_ASSERT(state->ops->dump_data_interface);

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

void starpu_load_cuda_function(int devid, struct starpu_cuda_function_s *function)
{
	CUresult res;

	/* load the module on the device if it is not already the case */
	starpu_load_cuda_module(devid, function->module);

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

void deinit_context(int devid)
{
	cublasShutdown();

	/* cleanup the runtime API internal stuffs (which CUBLAS is using) */
	status = cudaThreadExit();
	if (status)
		CUDA_REPORT_ERROR(status);

	/* XXX driver API and runtime API does not seem to like each other,
	 * so until CUDA is fixed, we cannot properly cleanup the cuInit that
	 * was done initially */
//	status = cuCtxDestroy(cuContext[devid]);
//	if (status)
//		CUDA_REPORT_ERROR(status);
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

	ncudagpus = get_cuda_device_count();
	assert(ncudagpus <= MAXCUDADEVS);
}

int execute_job_on_cuda(job_t j, struct worker_s *args, unsigned use_cublas)
{
	int ret;
//	uint32_t mask = (1<<0);
	uint32_t mask = 0;

	STARPU_ASSERT(j);
	struct starpu_task *task = j->task;

	CUresult status;
	tick_t codelet_start, codelet_end;
	tick_t codelet_start_comm, codelet_end_comm;
	
	int devid = args->id;

	unsigned calibrate_model = 0;

	STARPU_ASSERT(task);
	struct starpu_codelet_t *cl = task->cl;
	STARPU_ASSERT(cl);

	if (cl->model && cl->model->benchmarking) 
		calibrate_model = 1;

	/* we do not take communication into account when modeling the performance */
	if (calibrate_model || BENCHMARK_COMM)
	{
		cuCtxSynchronize();
		GET_TICK(codelet_start_comm);
	}

	ret = fetch_codelet_input(task->buffers, task->interface, cl->nbuffers, mask);
	if (ret != 0) {
		/* there was not enough memory, so the input of
		 * the codelet cannot be fetched ... put the 
		 * codelet back, and try it later */
		return STARPU_TRYAGAIN;
	}

	if (calibrate_model || BENCHMARK_COMM)
	{
		cuCtxSynchronize();
		GET_TICK(codelet_end_comm);
	}



	TRACE_START_CODELET_BODY(j);
	if (use_cublas) {
		cl_func func = cl->cublas_func;
		STARPU_ASSERT(func);
		GET_TICK(codelet_start);
		func(task->interface, task->cl_arg);
		cuCtxSynchronize();
		GET_TICK(codelet_end);
	} else {
		/* load the module and the function */
		starpu_cuda_codelet_t *args; 
		args = cl->cuda_func;

		starpu_load_cuda_function(devid, args->func);

		status = cuFuncSetBlockShape(args->func->function,
					args->blockx, 
					args->blocky, 1);
		if (status) {
			CUDA_REPORT_ERROR(status);
		}

		/* set up the function args */
		set_function_args(args, task->buffers, task->interface, cl->nbuffers);

		/* set up the grids */
//#ifdef MODEL_DEBUG
		if (calibrate_model || BENCHMARK_COMM)
		{
			status = cuCtxSynchronize();
			GET_TICK(codelet_start);
		}
//#endif
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

//#ifdef MODEL_DEBUG
	
	if (calibrate_model || BENCHMARK_COMM)
	{
		double measured = timing_delay(&codelet_start, &codelet_end);
		double measured_comm = timing_delay(&codelet_start_comm, &codelet_end_comm);

		args->jobq->total_computation_time += measured;
		args->jobq->total_communication_time += measured_comm;

		if (calibrate_model)
			update_perfmodel_history(j, args->arch, measured);
	}
//#endif

	push_codelet_output(task->buffers, cl->nbuffers, mask);

	return STARPU_SUCCESS;
}

void *cuda_worker(void *arg)
{
	struct worker_s* args = arg;

	int devid = args->id;
	unsigned memory_node = args->memory_node;

#ifdef USE_FXT
	fxt_register_thread(args->bindid);
#endif
	TRACE_NEW_WORKER(FUT_CUDA_KEY, memory_node);

	bind_thread_on_cpu(args->bindid);

	set_local_memory_node_key(&(args->memory_node));

	set_local_queue(args->jobq);

	/* this is only useful (and meaningful) is there is a single
	   memory node "related" to that queue */
	args->jobq->memory_node = memory_node;

	args->jobq->total_computation_time = 0.0;
	args->jobq->total_communication_time = 0.0;

	init_context(devid);
#ifdef VERBOSE
	fprintf(stderr, "cuda thread is ready to run on CPU %d !\n", args->bindid);
#endif

//	uint64_t foo = 1664;
//	cuMemAlloc(&debugptr, sizeof(uint64_t));
//	cuMemcpyHtoD(debugptr, &foo, sizeof(uint64_t));
	

	/* tell the main thread that this one is ready */
	pthread_mutex_lock(&args->mutex);
	args->worker_is_initialized = 1;
	pthread_cond_signal(&args->ready_cond);
	pthread_mutex_unlock(&args->mutex);

	struct job_s * j;
	int res;
	
	while (machine_is_running())
	{
		datawizard_progress(args->memory_node);

		//int debugfoo;
		j = pop_task();
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
		res = execute_job_on_cuda(j, args, use_cublas);

		if (res != STARPU_SUCCESS) {
			switch (res) {
				case STARPU_SUCCESS:
				case STARPU_FATAL:
					assert(0);
				case STARPU_TRYAGAIN:
					fprintf(stderr, "ouch, put the codelet %p back ... \n", j);
					push_task(j);
					STARPU_ASSERT(0);
					continue;
				default:
					assert(0);
			}
		}

		handle_job_termination(j);

//		cuMemcpyDtoH(&debugfoo, debugptr, sizeof(uint64_t));
//		printf("AFTER TASK, debug ptr = %p\n", debugfoo);
	} 

	deinit_context(devid);

#ifdef DATA_STATS
	fprintf(stderr, "CUDA #%d computation %le comm %le (%lf \%%)\n", args->id, args->jobq->total_computation_time, args->jobq->total_communication_time, args->jobq->total_communication_time*100.0/args->jobq->total_computation_time);
#endif
	pthread_exit(NULL);

	TRACE_WORKER_TERMINATED(FUT_CUDA_KEY);

	return NULL;

//error:
//	CUDA_REPORT_ERROR(status);
//	assert(0);

}
