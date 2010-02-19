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
#include "driver_cpu.h"
#include <core/policies/sched_policy.h>

static int execute_job_on_cpu(job_t j, struct worker_s *cpu_args)
{
	int ret;
	tick_t codelet_start, codelet_end;
	tick_t codelet_start_comm, codelet_end_comm;

	unsigned calibrate_model = 0;
	struct starpu_task *task = j->task;
	struct starpu_codelet_t *cl = task->cl;

	STARPU_ASSERT(cl);
	STARPU_ASSERT(cl->cpu_func);

	if (cl->model && cl->model->benchmarking)
		calibrate_model = 1;

	if (calibrate_model || BENCHMARK_COMM)
		GET_TICK(codelet_start_comm);

	ret = _starpu_fetch_task_input(task, 0);

	if (calibrate_model || BENCHMARK_COMM)
		GET_TICK(codelet_end_comm);

	if (ret != 0) {
		/* there was not enough memory so the codelet cannot be executed right now ... */
		/* push the codelet back and try another one ... */
		return -EAGAIN;
	}

	TRACE_START_CODELET_BODY(j);

	if (calibrate_model || BENCHMARK_COMM)
		GET_TICK(codelet_start);

	cpu_args->status = STATUS_EXECUTING;
	cl_func func = cl->cpu_func;
	func(task->interface, task->cl_arg);

	cl->per_worker_stats[cpu_args->workerid]++;
	
	if (calibrate_model || BENCHMARK_COMM)
		GET_TICK(codelet_end);

	TRACE_END_CODELET_BODY(j);
	cpu_args->status = STATUS_UNKNOWN;

	starpu_push_task_output(task, 0);

//#ifdef STARPU_MODEL_DEBUG
	if (calibrate_model || BENCHMARK_COMM)
	{
		double measured = timing_delay(&codelet_start, &codelet_end);
		double measured_comm = timing_delay(&codelet_start_comm, &codelet_end_comm);

//		fprintf(stderr, "%d\t%d\n", (int)j->penality, (int)measured_comm);
		cpu_args->jobq->total_computation_time += measured;
		cpu_args->jobq->total_communication_time += measured_comm;

		double error;
		error = fabs(STARPU_MAX(measured, 0.0) - STARPU_MAX(j->predicted, 0.0)); 
//		fprintf(stderr, "Error -> %le, predicted -> %le measured ->%le\n", error, j->predicted, measured);
		cpu_args->jobq->total_computation_time_error += error;

		if (calibrate_model)
			_starpu_update_perfmodel_history(j, cpu_args->arch, cpu_args->id, measured);
	}
//#endif

	cpu_args->jobq->total_job_performed++;

	return 0;
}

void *_starpu_cpu_worker(void *arg)
{
	struct worker_s *cpu_arg = arg;

#ifdef STARPU_USE_FXT
	fxt_register_thread(cpu_arg->bindid);
#endif
	TRACE_WORKER_INIT_START(FUT_CPU_KEY, cpu_arg->memory_node);

	_starpu_bind_thread_on_cpu(cpu_arg->config, cpu_arg->bindid);

#ifdef STARPU_VERBOSE
        fprintf(stderr, "cpu worker %d is ready on logical cpu %d\n", cpu_arg->id, cpu_arg->bindid);
#endif

	set_local_memory_node_key(&cpu_arg->memory_node);

	set_local_queue(cpu_arg->jobq);

	_starpu_set_local_worker_key(cpu_arg);

	snprintf(cpu_arg->name, 32, "CPU %d", cpu_arg->id);

	/* this is only useful (and meaningful) is there is a single
	   memory node "related" to that queue */
	cpu_arg->jobq->memory_node = cpu_arg->memory_node;

	cpu_arg->jobq->total_computation_time = 0.0;
	cpu_arg->jobq->total_communication_time = 0.0;
	cpu_arg->jobq->total_computation_time_error = 0.0;
	cpu_arg->jobq->total_job_performed = 0;

	cpu_arg->status = STATUS_UNKNOWN;
	
	TRACE_WORKER_INIT_END

        /* tell the main thread that we are ready */
	pthread_mutex_lock(&cpu_arg->mutex);
	cpu_arg->worker_is_initialized = 1;
	pthread_cond_signal(&cpu_arg->ready_cond);
	pthread_mutex_unlock(&cpu_arg->mutex);

        job_t j;
	int res;

	struct sched_policy_s *policy = get_sched_policy();
	struct jobq_s *queue = policy->get_local_queue(policy);
	unsigned memnode = cpu_arg->memory_node;

	while (_starpu_machine_is_running())
	{
		TRACE_START_PROGRESS(memnode);
		_starpu_datawizard_progress(memnode, 1);
		TRACE_END_PROGRESS(memnode);

		_starpu_execute_registered_progression_hooks();

		jobq_lock(queue);

		/* perhaps there is some local task to be executed first */
		j = _starpu_pop_local_task(cpu_arg);

		/* otherwise ask a task to the scheduler */
		if (!j)
			j = pop_task();

                if (j == NULL) {
			if (_starpu_worker_can_block(memnode))
				pthread_cond_wait(&queue->activity_cond, &queue->activity_mutex);
			jobq_unlock(queue);
 			continue;
		};
		
		jobq_unlock(queue);

		/* can a cpu perform that task ? */
		if (!STARPU_CPU_MAY_PERFORM(j)) 
		{
			/* put it and the end of the queue ... XXX */
			push_task(j);
			continue;
		}

                res = execute_job_on_cpu(j, cpu_arg);
		if (res) {
			switch (res) {
				case -EAGAIN:
					push_task(j);
					continue;
				default: 
					assert(0);
			}
		}

		_starpu_handle_job_termination(j);
        }

	TRACE_WORKER_DEINIT_START

#ifdef STARPU_DATA_STATS
	fprintf(stderr, "CPU #%d computation %le comm %le (%lf \%%)\n", cpu_arg->id, cpu_arg->jobq->total_computation_time, cpu_arg->jobq->total_communication_time,  cpu_arg->jobq->total_communication_time*100.0/cpu_arg->jobq->total_computation_time);
#endif

#ifdef STARPU_VERBOSE
	double ratio = 0;
	if (cpu_arg->jobq->total_job_performed != 0)
	{
		ratio = cpu_arg->jobq->total_computation_time_error/cpu_arg->jobq->total_computation_time;
	}

	_starpu_print_to_logfile("MODEL ERROR: CPU %d ERROR %lf EXEC %lf RATIO %lf NTASKS %d\n", cpu_arg->id, cpu_arg->jobq->total_computation_time_error, cpu_arg->jobq->total_computation_time, ratio, cpu_arg->jobq->total_job_performed);
#endif

	TRACE_WORKER_DEINIT_END(FUT_CPU_KEY);

	pthread_exit(NULL);
}
