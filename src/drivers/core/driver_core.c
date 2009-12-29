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

#include <math.h>

#include <core/debug.h>
#include "driver_core.h"
#include <core/policies/sched_policy.h>

static int execute_job_on_core(job_t j, struct worker_s *core_args)
{
	int ret;
	tick_t codelet_start, codelet_end;
	tick_t codelet_start_comm, codelet_end_comm;

	unsigned calibrate_model = 0;
	struct starpu_task *task = j->task;
	struct starpu_codelet_t *cl = task->cl;

	STARPU_ASSERT(cl);
	STARPU_ASSERT(cl->core_func);

	if (cl->model && cl->model->benchmarking)
		calibrate_model = 1;

	if (calibrate_model || BENCHMARK_COMM)
		GET_TICK(codelet_start_comm);

	ret = fetch_task_input(task, 0);

	if (calibrate_model || BENCHMARK_COMM)
		GET_TICK(codelet_end_comm);

	if (ret != 0) {
		/* there was not enough memory so the codelet cannot be executed right now ... */
		/* push the codelet back and try another one ... */
		return STARPU_TRYAGAIN;
	}

	TRACE_START_CODELET_BODY(j);

	if (calibrate_model || BENCHMARK_COMM)
		GET_TICK(codelet_start);

	core_args->status = STATUS_EXECUTING;
	cl_func func = cl->core_func;
	func(task->interface, task->cl_arg);

	cl->per_worker_stats[core_args->workerid]++;
	
	if (calibrate_model || BENCHMARK_COMM)
		GET_TICK(codelet_end);

	TRACE_END_CODELET_BODY(j);
	core_args->status = STATUS_UNKNOWN;

	push_task_output(task, 0);

//#ifdef MODEL_DEBUG
	if (calibrate_model || BENCHMARK_COMM)
	{
		double measured = timing_delay(&codelet_start, &codelet_end);
		double measured_comm = timing_delay(&codelet_start_comm, &codelet_end_comm);

//		fprintf(stderr, "%d\t%d\n", (int)j->penality, (int)measured_comm);
		core_args->jobq->total_computation_time += measured;
		core_args->jobq->total_communication_time += measured_comm;

		double error;
		error = fabs(STARPU_MAX(measured, 0.0) - STARPU_MAX(j->predicted, 0.0)); 
//		fprintf(stderr, "Error -> %le, predicted -> %le measured ->%le\n", error, j->predicted, measured);
		core_args->jobq->total_computation_time_error += error;

		if (calibrate_model)
			update_perfmodel_history(j, core_args->arch, core_args->id, measured);
	}
//#endif

	core_args->jobq->total_job_performed++;

	return STARPU_SUCCESS;
}

void *core_worker(void *arg)
{
	struct worker_s *core_arg = arg;

#ifdef USE_FXT
	fxt_register_thread(core_arg->bindid);
#endif
	TRACE_WORKER_INIT_START(FUT_CORE_KEY, core_arg->memory_node);

	bind_thread_on_cpu(core_arg->config, core_arg->bindid);

#ifdef VERBOSE
        fprintf(stderr, "core worker %d is ready on logical core %d\n", core_arg->id, core_arg->bindid);
#endif

	set_local_memory_node_key(&core_arg->memory_node);

	set_local_queue(core_arg->jobq);

	set_local_worker_key(core_arg);

	snprintf(core_arg->name, 32, "CORE %d", core_arg->id);

	/* this is only useful (and meaningful) is there is a single
	   memory node "related" to that queue */
	core_arg->jobq->memory_node = core_arg->memory_node;

	core_arg->jobq->total_computation_time = 0.0;
	core_arg->jobq->total_communication_time = 0.0;
	core_arg->jobq->total_computation_time_error = 0.0;
	core_arg->jobq->total_job_performed = 0;

	core_arg->status = STATUS_UNKNOWN;
	
	TRACE_WORKER_INIT_END

        /* tell the main thread that we are ready */
	pthread_mutex_lock(&core_arg->mutex);
	core_arg->worker_is_initialized = 1;
	pthread_cond_signal(&core_arg->ready_cond);
	pthread_mutex_unlock(&core_arg->mutex);

        job_t j;
	int res;

	struct sched_policy_s *policy = get_sched_policy();
	struct jobq_s *queue = policy->get_local_queue(policy);
	unsigned memnode = core_arg->memory_node;

	while (machine_is_running())
	{
		TRACE_START_PROGRESS(memnode);
		datawizard_progress(memnode, 1);
		TRACE_END_PROGRESS(memnode);

		jobq_lock(queue);

		/* perhaps there is some local task to be executed first */
		j = pop_local_task(core_arg);

		/* otherwise ask a task to the scheduler */
		if (!j)
			j = pop_task();

                if (j == NULL) {
			if (check_that_no_data_request_exists(memnode) && machine_is_running())
				pthread_cond_wait(&queue->activity_cond, &queue->activity_mutex);
			jobq_unlock(queue);
 			continue;
		};
		
		jobq_unlock(queue);

		/* can a core perform that task ? */
		if (!CORE_MAY_PERFORM(j)) 
		{
			/* put it and the end of the queue ... XXX */
			push_task(j);
			continue;
		}

                res = execute_job_on_core(j, core_arg);
		if (res != STARPU_SUCCESS) {
			switch (res) {
				case STARPU_FATAL:
					assert(0);
				case STARPU_TRYAGAIN:
					push_task(j);
					continue;
				default: 
					assert(0);
			}
		}

		handle_job_termination(j);
        }

	TRACE_WORKER_DEINIT_START

#ifdef DATA_STATS
	fprintf(stderr, "CORE #%d computation %le comm %le (%lf \%%)\n", core_arg->id, core_arg->jobq->total_computation_time, core_arg->jobq->total_communication_time,  core_arg->jobq->total_communication_time*100.0/core_arg->jobq->total_computation_time);
#endif

#ifdef VERBOSE
	double ratio = 0;
	if (core_arg->jobq->total_job_performed != 0)
	{
		ratio = core_arg->jobq->total_computation_time_error/core_arg->jobq->total_computation_time;
	}

	print_to_logfile("MODEL ERROR: CORE %d ERROR %lf EXEC %lf RATIO %lf NTASKS %d\n", core_arg->id, core_arg->jobq->total_computation_time_error, core_arg->jobq->total_computation_time, ratio, core_arg->jobq->total_job_performed);
#endif

	TRACE_WORKER_DEINIT_END(FUT_CORE_KEY);

	pthread_exit(NULL);
}
