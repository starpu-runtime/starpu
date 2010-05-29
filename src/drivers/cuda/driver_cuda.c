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

#include <common/utils.h>
#include <common/config.h>
#include <core/debug.h>
#include "driver_cuda.h"
#include <core/policies/sched_policy.h>

/* the number of CUDA devices */
static int ncudagpus;

static cudaStream_t streams[STARPU_NMAXWORKERS];

cudaStream_t *starpu_cuda_get_local_stream(void)
{
	int worker = starpu_worker_get_id();

	return &streams[worker];
}

static void init_context(int devid)
{
	cudaError_t cures;

	cures = cudaSetDevice(devid);
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);

	/* force CUDA to initialize the context for real */
	cudaFree(0);

	cures = cudaStreamCreate(starpu_cuda_get_local_stream());
	if (STARPU_UNLIKELY(cures))
		STARPU_CUDA_REPORT_ERROR(cures);
}

static void deinit_context(int workerid)
{
	cudaError_t cures;

	cudaStreamDestroy(streams[workerid]);

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
		 STARPU_CUDA_REPORT_ERROR(cures);
	
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
//	uint32_t mask = (1<<0);
	uint32_t mask = 0;

	STARPU_ASSERT(j);
	struct starpu_task *task = j->task;

	cudaError_t cures;
	starpu_tick_t codelet_start, codelet_end;
	starpu_tick_t codelet_start_comm, codelet_end_comm;
	
	unsigned calibrate_model = 0;

	STARPU_ASSERT(task);
	struct starpu_codelet_t *cl = task->cl;
	STARPU_ASSERT(cl);

	if (cl->model && cl->model->benchmarking) 
		calibrate_model = 1;

	/* we do not take communication into account when modeling the performance */
	if (STARPU_BENCHMARK_COMM)
	{
		cures = cudaThreadSynchronize();
		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);
		STARPU_GET_TICK(codelet_start_comm);
	}

	ret = _starpu_fetch_task_input(task, mask);
	if (ret != 0) {
		/* there was not enough memory, so the input of
		 * the codelet cannot be fetched ... put the 
		 * codelet back, and try it later */
		return -EAGAIN;
	}

	if (calibrate_model || STARPU_BENCHMARK_COMM)
	{
		cures = cudaThreadSynchronize();
		if (STARPU_UNLIKELY(cures))
			STARPU_CUDA_REPORT_ERROR(cures);
		STARPU_GET_TICK(codelet_end_comm);
	}

	STARPU_TRACE_START_CODELET_BODY(j);

	args->status = STATUS_EXECUTING;
	cl_func func = cl->cuda_func;
	STARPU_ASSERT(func);
	STARPU_GET_TICK(codelet_start);
	func(task->interface, task->cl_arg);

	cl->per_worker_stats[args->workerid]++;

	STARPU_GET_TICK(codelet_end);

	args->status = STATUS_UNKNOWN;

	STARPU_TRACE_END_CODELET_BODY(j);	

	if (calibrate_model || STARPU_BENCHMARK_COMM)
	{
		double measured = _starpu_timing_delay(&codelet_start, &codelet_end);
		double measured_comm = _starpu_timing_delay(&codelet_start_comm, &codelet_end_comm);

		args->jobq->total_computation_time += measured;
		args->jobq->total_communication_time += measured_comm;

		double error;
		error = fabs(STARPU_MAX(measured, 0.0) - STARPU_MAX(j->predicted, 0.0)); 
		args->jobq->total_computation_time_error += error;

		if (calibrate_model)
			_starpu_update_perfmodel_history(j, args->perf_arch, (unsigned)args->devid, measured);
	}

	STARPU_ATOMIC_ADD(&args->jobq->total_job_performed, 1);

	_starpu_push_task_output(task, mask);

	return 0;
}

void *_starpu_cuda_worker(void *arg)
{
	struct starpu_worker_s* args = arg;
	struct starpu_jobq_s *jobq = args->jobq;

	int devid = args->devid;
	unsigned memory_node = args->memory_node;

#ifdef STARPU_USE_FXT
	_starpu_fxt_register_thread(args->bindid);
#endif
	STARPU_TRACE_WORKER_INIT_START(STARPU_FUT_CUDA_KEY, memory_node);

	_starpu_bind_thread_on_cpu(args->config, args->bindid);

	_starpu_set_local_memory_node_key(&(args->memory_node));

	_starpu_set_local_queue(jobq);

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
	snprintf(args->name, 32, "CUDA %d (%s)", args->devid, devname);

#ifdef STARPU_VERBOSE
	fprintf(stderr, "cuda (%s) dev id %d thread is ready to run on CPU %d !\n", devname, devid, args->bindid);
#endif

	STARPU_TRACE_WORKER_INIT_END

	/* tell the main thread that this one is ready */
	PTHREAD_MUTEX_LOCK(&args->mutex);
	args->worker_is_initialized = 1;
	PTHREAD_COND_SIGNAL(&args->ready_cond);
	PTHREAD_MUTEX_UNLOCK(&args->mutex);

	struct starpu_job_s * j;
	int res;

	struct starpu_sched_policy_s *policy = _starpu_get_sched_policy();
	struct starpu_jobq_s *queue = policy->starpu_get_local_queue(policy);
	unsigned memnode = args->memory_node;
	
	while (_starpu_machine_is_running())
	{
		STARPU_TRACE_START_PROGRESS(memnode);
		_starpu_datawizard_progress(memnode, 1);
		STARPU_TRACE_END_PROGRESS(memnode);

		_starpu_execute_registered_progression_hooks();
	
		_starpu_jobq_lock(queue);

		/* perhaps there is some local task to be executed first */
		j = _starpu_pop_local_task(args);

		/* otherwise ask a task to the scheduler */
		if (!j)
			j = _starpu_pop_task();

		if (j == NULL) {
			if (_starpu_worker_can_block(memnode))
				PTHREAD_COND_WAIT(&queue->activity_cond, &queue->activity_mutex);
			_starpu_jobq_unlock(queue);
			continue;
		}

		_starpu_jobq_unlock(queue);

		/* can CUDA do that task ? */
		if (!STARPU_CUDA_MAY_PERFORM(j))
		{
			/* this is neither a cuda or a cublas task */
			_starpu_push_task(j, 0);
			continue;
		}

		_starpu_set_current_task(j->task);

		res = execute_job_on_cuda(j, args);

		_starpu_set_current_task(NULL);

		if (res) {
			switch (res) {
				case -EAGAIN:
					fprintf(stderr, "ouch, put the codelet %p back ... \n", j);
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

	/* In case there remains some memory that was automatically
	 * allocated by StarPU, we release it now. Note that data
	 * coherency is not maintained anymore at that point ! */
	_starpu_free_all_automatically_allocated_buffers(memnode);

	deinit_context(args->workerid);

#ifdef STARPU_DATA_STATS
	fprintf(stderr, "CUDA #%d computation %le comm %le (%lf \%%)\n", args->id, jobq->total_computation_time, jobq->total_communication_time, jobq->total_communication_time*100.0/jobq->total_computation_time);
#endif

#ifdef STARPU_VERBOSE
	double ratio = 0;
	if (jobq->total_job_performed != 0)
	{
		ratio = jobq->total_computation_time_error/jobq->total_computation_time;
	}


	_starpu_print_to_logfile("MODEL ERROR: CUDA %d ERROR %lf EXEC %lf RATIO %lf NTASKS %d\n", args->devid, jobq->total_computation_time_error, jobq->total_computation_time, ratio, jobq->total_job_performed);
#endif

	STARPU_TRACE_WORKER_DEINIT_END(STARPU_FUT_CUDA_KEY);

	pthread_exit(NULL);

	return NULL;

}
