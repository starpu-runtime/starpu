/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010, 2011  Université de Bordeaux 1
 * Copyright (C) 2010  Mehdi Juhoor <mjuhoor@gmail.com>
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
 * Copyright (C) 2011  Télécom-SudParis
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#include <starpu_cuda.h>
#include <common/utils.h>
#include <common/config.h>
#include <core/debug.h>
#include <drivers/driver_common/driver_common.h>
#include "driver_cuda.h"
#include <core/sched_policy.h>


double _starpu_task_get_conversion_time(struct starpu_task *task);

/* the number of CUDA devices */
static int ncudagpus;

static cudaStream_t streams[STARPU_NMAXWORKERS];
static cudaStream_t transfer_streams[STARPU_NMAXWORKERS];

/* In case we want to cap the amount of memory available on the GPUs by the
 * mean of the STARPU_LIMIT_GPU_MEM, we allocate a big buffer when the driver
 * is launched. */
static char *wasted_memory[STARPU_NMAXWORKERS];

static void limit_gpu_mem_if_needed(int devid)
{
	cudaError_t cures;
	int limit = starpu_get_env_number("STARPU_LIMIT_GPU_MEM");

	if (limit == -1)
	{
		wasted_memory[devid] = NULL;
		return;
	}

	/* Find the size of the memory on the device */
	struct cudaDeviceProp prop;
	cures = cudaGetDeviceProperties(&prop, devid);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	size_t totalGlobalMem = prop.totalGlobalMem;

	/* How much memory to waste ? */
	size_t to_waste = totalGlobalMem - (size_t)limit*1024*1024;

	_STARPU_DEBUG("CUDA device %d: Wasting %ld MB / Limit %ld MB / Total %ld MB / Remains %ld MB\n",
			devid, (size_t)to_waste/(1024*1024), (size_t)limit, (size_t)totalGlobalMem/(1024*1024),
			(size_t)(totalGlobalMem - to_waste)/(1024*1024));
	
	/* Allocate a large buffer to waste memory and constraint the amount of available memory. */
	cures = cudaMalloc((void **)&wasted_memory[devid], to_waste);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);
}

static void unlimit_gpu_mem_if_needed(int devid)
{
	cudaError_t cures;

	if (wasted_memory[devid])
	{
		cures = cudaFree(wasted_memory[devid]);
		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);

		wasted_memory[devid] = NULL;
	}
}

size_t starpu_cuda_get_global_mem_size(int devid)
{
	cudaError_t cures;
	struct cudaDeviceProp prop;

	/* Find the size of the memory on the device */
	cures = cudaGetDeviceProperties(&prop, devid);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	return (size_t)prop.totalGlobalMem;
}

cudaStream_t starpu_cuda_get_local_transfer_stream(void)
{
	int worker = starpu_worker_get_id();

	return transfer_streams[worker];
}

cudaStream_t starpu_cuda_get_local_stream(void)
{
	int worker = starpu_worker_get_id();

	return streams[worker];
}

static void init_context(int devid)
{
	cudaError_t cures;
	int workerid = starpu_worker_get_id();

	cures = cudaSetDevice(devid);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	/* force CUDA to initialize the context for real */
	cudaFree(0);

	limit_gpu_mem_if_needed(devid);

	cures = cudaStreamCreate(&streams[workerid]);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	cures = cudaStreamCreate(&transfer_streams[workerid]);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);
}

static void deinit_context(int workerid, int devid)
{
	cudaError_t cures;

	cudaStreamDestroy(streams[workerid]);
	cudaStreamDestroy(transfer_streams[workerid]);

	unlimit_gpu_mem_if_needed(devid);

	/* cleanup the runtime API internal stuffs (which CUBLAS is using) */
	cures = cudaThreadExit();
	if (cures)
		STARPU_CUDA_REPORT_ERROR(cures);
}

unsigned _starpu_get_cuda_device_count(void)
{
	int cnt;

	cudaError_t cures;
	cures = cudaGetDeviceCount(&cnt);
	if (STARPU_UNLIKELY(cures))
		 return 0;

	if (cnt > STARPU_MAXCUDADEVS) {
		fprintf(stderr, "# Warning: %d CUDA devices available. Only %d enabled. Use configure option --enable-maxcudadev=xxx to update the maximum value of supported CUDA devices.\n", cnt, STARPU_MAXCUDADEVS);
		cnt = STARPU_MAXCUDADEVS;
	}
	return (unsigned)cnt;
}

void _starpu_init_cuda(void)
{
	ncudagpus = _starpu_get_cuda_device_count();
	assert(ncudagpus <= STARPU_MAXCUDADEVS);
}

static int execute_job_on_cuda(starpu_job_t j, struct starpu_worker_s *args)
{
	int ret;
	uint32_t mask = 0;
	cudaError_t cures;

	STARPU_ASSERT(j);
	struct starpu_task *task = j->task;

	struct timespec codelet_start, codelet_end;

	unsigned calibrate_model = 0;

	STARPU_ASSERT(task);
	struct starpu_codelet *cl = task->cl;
	STARPU_ASSERT(cl);

	if (cl->model && cl->model->benchmarking) 
		calibrate_model = 1;

	ret = _starpu_fetch_task_input(task, mask);
	if (ret != 0) {
		/* there was not enough memory, so the input of
		 * the codelet cannot be fetched ... put the 
		 * codelet back, and try it later */
		return -EAGAIN;
	}

	double conversion_time = _starpu_task_get_conversion_time(task);
	if (calibrate_model)
	{
		cures = cudaStreamSynchronize(starpu_cuda_get_local_transfer_stream());
		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);
	}

	_starpu_driver_start_job(args, j, &codelet_start, 0);

#ifdef HAVE_CUDA_MEMCPY_PEER
	/* We make sure we do manipulate the proper device */
	cures = cudaSetDevice(args->devid);
	if (STARPU_UNLIKELY(cures != cudaSuccess))
		STARPU_CUDA_REPORT_ERROR(cures);
#endif

	if (cl->cuda_func != STARPU_MULTIPLE_CUDA_IMPLEMENTATIONS) {
		cl_func func = cl->cuda_func;
		STARPU_ASSERT(func);
		func(task->interfaces, task->cl_arg);
	}
	else {
		/* _STARPU_DEBUG("Cuda driver : running kernel * (%d)\n", j->nimpl); */
		cl_func func = cl->cuda_funcs[j->nimpl];
		STARPU_ASSERT(func);
		func(task->interfaces, task->cl_arg);
	}

	_starpu_driver_end_job(args, j, &codelet_end, 0);

	_starpu_driver_update_job_feedback(j, args, args->perf_arch, &codelet_start, &codelet_end, conversion_time);

	_starpu_push_task_output(task, mask);

	return 0;
}

void *_starpu_cuda_worker(void *arg)
{
	struct starpu_worker_s* args = arg;

	int devid = args->devid;
	int workerid = args->workerid;
	unsigned memnode = args->memory_node;

#ifdef STARPU_USE_FXT
	_starpu_fxt_register_thread(args->bindid);
#endif
	STARPU_TRACE_WORKER_INIT_START(STARPU_FUT_CUDA_KEY, devid, memnode);

	_starpu_bind_thread_on_cpu(args->config, args->bindid);

	_starpu_set_local_memory_node_key(&memnode);

	_starpu_set_local_worker_key(args);

	init_context(devid);

	/* one more time to avoid hacks from third party lib :) */
	_starpu_bind_thread_on_cpu(args->config, args->bindid);

	args->status = STATUS_UNKNOWN;

	/* get the device's name */
	char devname[128];
	struct cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, devid);
	strncpy(devname, prop.name, 128);
	float size = (float) prop.totalGlobalMem / (1<<30);

#ifdef STARPU_HAVE_BUSID
#ifdef STARPU_HAVE_DOMAINID
	if (prop.pciDomainID)
		snprintf(args->name, sizeof(args->name), "CUDA %d (%s %.1f GiB %04x:%02x:%02x.0)", args->devid, devname, size, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
	else
#endif
		snprintf(args->name, sizeof(args->name), "CUDA %d (%s %.1f GiB %02x:%02x.0)", args->devid, devname, size, prop.pciBusID, prop.pciDeviceID);
#else
	snprintf(args->name, sizeof(args->name), "CUDA %d (%s %.1f GiB)", args->devid, devname, size);
#endif
	snprintf(args->short_name, sizeof(args->short_name), "CUDA %d", args->devid);
	_STARPU_DEBUG("cuda (%s) dev id %d thread is ready to run on CPU %d !\n", devname, devid, args->bindid);

	STARPU_TRACE_WORKER_INIT_END

	/* tell the main thread that this one is ready */
	_STARPU_PTHREAD_MUTEX_LOCK(&args->mutex);
	args->worker_is_initialized = 1;
	_STARPU_PTHREAD_COND_SIGNAL(&args->ready_cond);
	_STARPU_PTHREAD_MUTEX_UNLOCK(&args->mutex);

	struct starpu_job_s * j;
	struct starpu_task *task;
	int res;

	while (_starpu_machine_is_running())
	{
		STARPU_TRACE_START_PROGRESS(memnode);
		_starpu_datawizard_progress(memnode, 1);
		STARPU_TRACE_END_PROGRESS(memnode);

		_STARPU_PTHREAD_MUTEX_LOCK(args->sched_mutex);

		task = _starpu_pop_task(args);
	
                if (task == NULL) 
		{
			if (_starpu_worker_can_block(memnode))
				_starpu_block_worker(workerid, args->sched_cond, args->sched_mutex);

			_STARPU_PTHREAD_MUTEX_UNLOCK(args->sched_mutex);

			continue;
		};

		_STARPU_PTHREAD_MUTEX_UNLOCK(args->sched_mutex);

		STARPU_ASSERT(task);
		j = _starpu_get_job_associated_to_task(task);

		/* can CUDA do that task ? */
		if (!STARPU_CUDA_MAY_PERFORM(j))
		{
			/* this is neither a cuda or a cublas task */
			_starpu_push_task(j, 0);
			continue;
		}

		_starpu_set_current_task(task);

		res = execute_job_on_cuda(j, args);

		_starpu_set_current_task(NULL);

		if (res) {
			switch (res) {
				case -EAGAIN:
					_STARPU_DISP("ouch, put the codelet %p back ... \n", j);
					_starpu_push_task(j, 0);
					STARPU_ABORT();
					continue;
				default:
					assert(0);
			}
		}

		_starpu_handle_job_termination(j, 0);
	}

	STARPU_TRACE_WORKER_DEINIT_START

	_starpu_handle_all_pending_node_data_requests(memnode);

	/* In case there remains some memory that was automatically
	 * allocated by StarPU, we release it now. Note that data
	 * coherency is not maintained anymore at that point ! */
	_starpu_free_all_automatically_allocated_buffers(memnode);

	deinit_context(args->workerid, args->devid);

	STARPU_TRACE_WORKER_DEINIT_END(STARPU_FUT_CUDA_KEY);

	pthread_exit(NULL);

	return NULL;

}

void starpu_cublas_report_error(const char *func, const char *file, int line, cublasStatus status)
{
	char *errormsg;
	switch (status) {
		case CUBLAS_STATUS_SUCCESS:
			errormsg = "success";
			break;
		case CUBLAS_STATUS_NOT_INITIALIZED:
			errormsg = "not initialized";
			break;
		case CUBLAS_STATUS_ALLOC_FAILED:
			errormsg = "alloc failed";
			break;
		case CUBLAS_STATUS_INVALID_VALUE:
			errormsg = "invalid value";
			break;
		case CUBLAS_STATUS_ARCH_MISMATCH:
			errormsg = "arch mismatch";
			break;
		case CUBLAS_STATUS_EXECUTION_FAILED:
			errormsg = "execution failed";
			break;
		case CUBLAS_STATUS_INTERNAL_ERROR:
			errormsg = "internal error";
			break;
		default:
			errormsg = "unknown error";
			break;
	}
	printf("oops in %s (%s:%u)... %s \n", func, file, line, errormsg);
	assert(0);
}

void starpu_cuda_report_error(const char *func, const char *file, int line, cudaError_t status)
{
	const char *errormsg = cudaGetErrorString(status);
	printf("oops in %s (%s:%u)... %s \n", func, file, line, errormsg);
	assert(0);
}
