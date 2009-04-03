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

#include "driver_core.h"
#include <core/policies/sched_policy.h>

int execute_job_on_core(job_t j, struct worker_s *core_args)
{
	int ret;
	tick_t codelet_start, codelet_end;
	tick_t codelet_start_comm, codelet_end_comm;

	unsigned calibrate_model = 0;
	struct starpu_task *task = j->task;

	STARPU_ASSERT(task->cl);
	STARPU_ASSERT(task->cl->core_func);

	if (task->cl->model && task->cl->model->benchmarking)
		calibrate_model = 1;

	if (calibrate_model || BENCHMARK_COMM)
		GET_TICK(codelet_start_comm);

	ret = fetch_codelet_input(task->buffers, task->interface,
			task->cl->nbuffers, 0);

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

	cl_func func = task->cl->core_func;
	func(task->interface, task->cl_arg);
	
	if (calibrate_model || BENCHMARK_COMM)
		GET_TICK(codelet_end);

	TRACE_END_CODELET_BODY(j);

	push_codelet_output(task->buffers, task->cl->nbuffers, 0);

//#ifdef MODEL_DEBUG
	if (calibrate_model || BENCHMARK_COMM)
	{
		double measured = timing_delay(&codelet_start, &codelet_end);
		double measured_comm = timing_delay(&codelet_start_comm, &codelet_end_comm);

//		fprintf(stderr, "%d\t%d\n", (int)j->penality, (int)measured_comm);
		core_args->jobq->total_computation_time += measured;
		core_args->jobq->total_communication_time += measured_comm;

		if (calibrate_model)
			update_perfmodel_history(j, core_args->arch, measured);
	}
//#endif

	return STARPU_SUCCESS;
}

void *core_worker(void *arg)
{
	struct worker_s *core_arg = arg;

#ifdef USE_FXT
	fxt_register_thread(core_arg->bindid);
#endif
	TRACE_NEW_WORKER(FUT_CORE_KEY, core_arg->memory_node);

#ifndef DONTBIND
	/* fix the thread on the correct cpu */
	cpu_set_t aff_mask; 
	CPU_ZERO(&aff_mask);
	CPU_SET(core_arg->bindid, &aff_mask);
	sched_setaffinity(0, sizeof(aff_mask), &aff_mask);
#endif

#ifdef VERBOSE
        fprintf(stderr, "core worker %d is ready on logical core %d\n", core_arg->id, core_arg->bindid);
#endif

	set_local_memory_node_key(&core_arg->memory_node);

	set_local_queue(core_arg->jobq);

	/* this is only useful (and meaningful) is there is a single
	   memory node "related" to that queue */
	core_arg->jobq->memory_node = core_arg->memory_node;

	core_arg->jobq->total_computation_time = 0.0;
	core_arg->jobq->total_communication_time = 0.0;
	
        /* tell the main thread that we are ready */
	pthread_mutex_lock(&core_arg->mutex);
	core_arg->worker_is_initialized = 1;
	pthread_cond_signal(&core_arg->ready_cond);
	pthread_mutex_unlock(&core_arg->mutex);

        job_t j;
	int res;

	while (machine_is_running())
	{
                j = pop_task();
                if (j == NULL) continue;

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

#ifdef DATA_STATS
	fprintf(stderr, "CORE #%d computation %le comm %le (%lf \%%)\n", core_arg->id, core_arg->jobq->total_computation_time, core_arg->jobq->total_communication_time,  core_arg->jobq->total_communication_time*100.0/core_arg->jobq->total_computation_time);
#endif

	TRACE_WORKER_TERMINATED(FUT_CORE_KEY);

	pthread_exit(NULL);
}
