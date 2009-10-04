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

#include <common/config.h>
#include <core/debug.h>
#include "driver_cuda.h"
#include <core/policies/sched_policy.h>

/* the number of CUDA devices */
static int ncudagpus;

static void init_context(int devid)
{
	cudaError_t cures;

	cures = cudaSetDevice(devid);
	if (STARPU_UNLIKELY(cures))
		CUDA_REPORT_ERROR(cures);
}

static void deinit_context(void)
{
	cudaError_t cures;

	/* cleanup the runtime API internal stuffs (which CUBLAS is using) */
	cures = cudaThreadExit();
	if (cures)
		CUDA_REPORT_ERROR(cures);
}

unsigned get_cuda_device_count(void)
{
	int cnt;

	cudaError_t cures;
	cures = cudaGetDeviceCount(&cnt);
	if (STARPU_UNLIKELY(cures))
		 CUDA_REPORT_ERROR(cures);
	
	return (unsigned)cnt;
}

void init_cuda(void)
{
	ncudagpus = get_cuda_device_count();
	assert(ncudagpus <= MAXCUDADEVS);
}

int execute_job_on_cuda(job_t j, struct worker_s *args)
{
	int ret;
//	uint32_t mask = (1<<0);
	uint32_t mask = 0;

	STARPU_ASSERT(j);
	struct starpu_task *task = j->task;

	cudaError_t cures;
	tick_t codelet_start, codelet_end;
	tick_t codelet_start_comm, codelet_end_comm;
	
	unsigned calibrate_model = 0;

	STARPU_ASSERT(task);
	struct starpu_codelet_t *cl = task->cl;
	STARPU_ASSERT(cl);

	if (cl->model && cl->model->benchmarking) 
		calibrate_model = 1;

	/* we do not take communication into account when modeling the performance */
	if (calibrate_model || BENCHMARK_COMM)
	{
		cures = cudaThreadSynchronize();
		if (STARPU_UNLIKELY(cures))
			CUDA_REPORT_ERROR(cures);
		GET_TICK(codelet_start_comm);
	}

	ret = fetch_task_input(task, mask);
	if (ret != 0) {
		/* there was not enough memory, so the input of
		 * the codelet cannot be fetched ... put the 
		 * codelet back, and try it later */
		return STARPU_TRYAGAIN;
	}

	if (calibrate_model || BENCHMARK_COMM)
	{
		cures = cudaThreadSynchronize();
		if (STARPU_UNLIKELY(cures))
			CUDA_REPORT_ERROR(cures);
		GET_TICK(codelet_end_comm);
	}

	unsigned memnode = get_local_memory_node();

	TRACE_START_CODELET_BODY(j);

	cl_func func = cl->cuda_func;
	STARPU_ASSERT(func);
	GET_TICK(codelet_start);
	func(task->interface, task->cl_arg);

	task->cl->per_worker_stats[args->workerid]++;

#if 0
	/* perform a barrier after the kernel */
	cures = cudaThreadSynchronize();
	if (STARPU_UNLIKELY(cures))
		CUDA_REPORT_ERROR(cures);
#endif

	GET_TICK(codelet_end);

	TRACE_END_CODELET_BODY(j);	

//#ifdef MODEL_DEBUG
	
	if (calibrate_model || BENCHMARK_COMM)
	{
		double measured = timing_delay(&codelet_start, &codelet_end);
		double measured_comm = timing_delay(&codelet_start_comm, &codelet_end_comm);

		args->jobq->total_computation_time += measured;
		args->jobq->total_communication_time += measured_comm;

		double error;
		error = fabs(STARPU_MAX(measured, 0.0) - STARPU_MAX(j->predicted, 0.0)); 
		args->jobq->total_computation_time_error += error;

		if (calibrate_model)
			update_perfmodel_history(j, args->perf_arch, (unsigned)args->id, measured);
	}
//#endif

	args->jobq->total_job_performed++;

	push_task_output(task, mask);

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

	bind_thread_on_cpu(args->config, args->bindid);

	set_local_memory_node_key(&(args->memory_node));

	set_local_queue(args->jobq);

	set_local_worker_key(args);

	/* this is only useful (and meaningful) is there is a single
	   memory node "related" to that queue */
	args->jobq->memory_node = memory_node;

	args->jobq->total_computation_time = 0.0;
	args->jobq->total_communication_time = 0.0;
	args->jobq->total_computation_time_error = 0.0;
	args->jobq->total_job_performed = 0;

	init_context(devid);

	/* get the device's name */
	char devname[128];
	struct cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, devid);
	strncpy(devname, prop.name, 128);
	snprintf(args->name, 32, "CUDA %d (%s)", args->id, devname);

#ifdef VERBOSE
	fprintf(stderr, "cuda (%s) thread is ready to run on CPU %d !\n", devname, args->bindid);
#endif

	/* tell the main thread that this one is ready */
	pthread_mutex_lock(&args->mutex);
	args->worker_is_initialized = 1;
	pthread_cond_signal(&args->ready_cond);
	pthread_mutex_unlock(&args->mutex);

	struct job_s * j;
	int res;

	struct sched_policy_s *policy = get_sched_policy();
	struct jobq_s *queue = policy->get_local_queue(policy);
	unsigned memnode = args->memory_node;
	
	while (machine_is_running())
	{
		TRACE_START_PROGRESS(memnode);
		datawizard_progress(memnode, 1);
		TRACE_END_PROGRESS(memnode);
	
		jobq_lock(queue);

		/* perhaps there is some local task to be executed first */
		j = pop_local_task(args);

		/* otherwise ask a task to the scheduler */
		if (!j)
			j = pop_task();

		if (j == NULL) {
			if (check_that_no_data_request_exists(memnode) && machine_is_running())
				pthread_cond_wait(&queue->activity_cond, &queue->activity_mutex);
			jobq_unlock(queue);
			continue;
		}

		jobq_unlock(queue);

		/* can CUDA do that task ? */
		if (!CUDA_MAY_PERFORM(j) && !CUBLAS_MAY_PERFORM(j))
		{
			/* this is neither a cuda or a cublas task */
			push_task(j);
			continue;
		}

		res = execute_job_on_cuda(j, args);

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
	}

	deinit_context();

#ifdef DATA_STATS
	fprintf(stderr, "CUDA #%d computation %le comm %le (%lf \%%)\n", args->id, args->jobq->total_computation_time, args->jobq->total_communication_time, args->jobq->total_communication_time*100.0/args->jobq->total_computation_time);
#endif

#ifdef VERBOSE
	double ratio = 0;
	if (args->jobq->total_job_performed != 0)
	{
		ratio = args->jobq->total_computation_time_error/args->jobq->total_computation_time;
	}


	print_to_logfile("MODEL ERROR: CUDA %d ERROR %lf EXEC %lf RATIO %lf NTASKS %d\n", args->id, args->jobq->total_computation_time_error, args->jobq->total_computation_time, ratio, args->jobq->total_job_performed);
#endif

	TRACE_WORKER_TERMINATED(FUT_CUDA_KEY);

	pthread_exit(NULL);

	return NULL;

}
