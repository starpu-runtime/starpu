/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2026  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu_config.h>
#include <starpu_scheduler.h>
#include <schedulers/starpu_scheduler_toolbox.h>
#include <schedulers/starpu_hpbasic.h>
#include <core/task.h>
#include <sched_policies/fifo_queues.h>
#include <limits.h>
#include <core/workers.h>
#include <datawizard/memory_nodes.h>

#ifdef HAVE_AYUDAME_H
#include <Ayudame.h>
#endif

#ifndef DBL_MIN
#define DBL_MIN __DBL_MIN__
#endif

#ifndef DBL_MAX
#define DBL_MAX __DBL_MAX__
#endif

/* Each type of task will have a different queue, sorted in decreaing order of CPU affinity */
struct _starpu_hpbasic_task_queue
{
	/* codelet would be unique id for each type of tasks*/
	struct starpu_codelet *task_codelet;
	/* queue for each type of tasks*/
	struct starpu_st_fifo_taskq *tasks_queue;
	/* accelration factor for tasks of this queue (CPU time / GPU time)*/
	double acceleration_factor;
};

struct _starpu_hpbasic_ready_queue
{
	unsigned int types_of_tasks;
	struct _starpu_hpbasic_task_queue task_queues[STARPU_HPBASIC_MAXTYPESOFTASKS];
	starpu_pthread_mutex_t hp_mutex;
};

static void init_hp_sched(unsigned sched_ctx_id)
{
	struct _starpu_hpbasic_ready_queue *data = (struct _starpu_hpbasic_ready_queue*)malloc(sizeof(struct _starpu_hpbasic_ready_queue));
	data->types_of_tasks = 0;

	/* Create a condition variable to protect it */

	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)data);

	starpu_pthread_mutex_init(&data->hp_mutex, NULL);
	_STARPU_DEBUG("Initialising hp scheduler\n");
}

static void deinit_hp_sched(unsigned sched_ctx_id)
{
	struct _starpu_hpbasic_ready_queue *data = (struct _starpu_hpbasic_ready_queue*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned queueId;
	for(queueId=0; queueId <data->types_of_tasks; queueId++)
	{
		STARPU_ASSERT(starpu_st_fifo_taskq_empty(data->task_queues[queueId].tasks_queue) != 0);
		starpu_st_fifo_taskq_destroy(data->task_queues[queueId].tasks_queue);
	}

	starpu_pthread_mutex_destroy(&data->hp_mutex);
	free(data);

	_STARPU_DEBUG("Destroying hp scheduler\n");
}

static double compute_acceleration_ratio(struct starpu_task *task, unsigned sched_ctx_id)
{
	/******* Calculation of max(CPU time) / min(GPU time) ******/
	double cpu_time = DBL_MIN;
	double gpu_time = DBL_MAX;
	unsigned impl_mask;
	unsigned worker, nimpl;

	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);

	struct starpu_sched_ctx_iterator it;
	workers->init_iterator_for_parallel_tasks(workers, &it, task);
	while(workers->has_next(workers, &it))
	{
		worker = workers->get_next(workers, &it);
		struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(worker, sched_ctx_id);
		if (!starpu_worker_can_execute_task_impl(worker, task, &impl_mask))
			continue;

		if(starpu_worker_get_type(worker) == STARPU_CPU_WORKER)
		{
			for (nimpl  = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
			{
				if (!(impl_mask & (1U << nimpl)))
				{
					/* no one on that queue may execute this task */
					continue;
				}
				double expected_length = starpu_task_expected_length(task, perf_arch, nimpl);
				if(expected_length > cpu_time)
					cpu_time = expected_length;
			}
		}
		else if(starpu_worker_get_type(worker) == STARPU_CUDA_WORKER)
		{
			for (nimpl  = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
			{
				if (!(impl_mask & (1U << nimpl)))
				{
					/* no one on that queue may execute this task */
					continue;
				}
				double expected_length = starpu_task_expected_length(task, perf_arch, nimpl);
				if(expected_length < gpu_time)
					gpu_time = expected_length;
			}
		}
		else
			/* other than CPU/GPU worker */
			continue;
	}
	/*To avoid division by zero in some undesirable situation
	 * add 1 to denominator
	 *
	 */
	return cpu_time/(gpu_time + 1.0);
}

static int push_task_in_to_ready_queue(struct starpu_task *task)
{
	STARPU_ASSERT_MSG((task->cl->where & STARPU_CPU) && (task->cl->where & STARPU_CUDA), "Some tasks do not have both CPU and GPU implementations");

	unsigned sched_ctx_id = task->sched_ctx;
	struct _starpu_hpbasic_ready_queue *data = (struct _starpu_hpbasic_ready_queue*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_pthread_mutex_lock(&data->hp_mutex);
	/* Insert task in the appropriate queue */
	unsigned int queueId;
	for(queueId=0; queueId<data->types_of_tasks; queueId++)
	{
		/* checking whether other tasks with the same codelt have been
		 * recorded in the past or not */
		/************** pointer comparison **********/
		if(data->task_queues[queueId].task_codelet == task->cl)
		{
			//TODO: add tasks in nonincreaing order of priorities
			starpu_st_fifo_taskq_push_back_task(data->task_queues[queueId].tasks_queue, task);
			break;
		}
	}
	if(queueId == data->types_of_tasks)
	{
		//first instance of this type of task
		/* Create a new ready queue to store this type of tasks */
		struct starpu_st_fifo_taskq *current_taskq = starpu_st_fifo_taskq_create();
		starpu_st_fifo_taskq_push_back_task(current_taskq, task);
		//compute acceleration factor for this type of tasks
		double current_acceleration_factor = compute_acceleration_ratio(task, sched_ctx_id);
		/* insert the queues based on the nondecreasing order of acceleration factor */
		/* Insertion sort based on acceleration factor */
		while((queueId > 0) && (data->task_queues[queueId-1].acceleration_factor > current_acceleration_factor))
		{
			data->task_queues[queueId].task_codelet = data->task_queues[queueId - 1].task_codelet;
			data->task_queues[queueId].tasks_queue = data->task_queues[queueId - 1].tasks_queue;
			data->task_queues[queueId].acceleration_factor = data->task_queues[queueId - 1].acceleration_factor;
			queueId--;
		}
		/* pointer to task codelet */
		data->task_queues[queueId].task_codelet = task->cl;
		data->task_queues[queueId].tasks_queue = current_taskq;
		data->task_queues[queueId].acceleration_factor = current_acceleration_factor;

		data->types_of_tasks ++;
	}

	starpu_push_task_end(task);
	starpu_pthread_mutex_unlock(&data->hp_mutex);

#ifndef STARPU_NON_BLOCKING_DRIVERS
	/*if there are no tasks block */
	/* wake people waiting for a task */
	unsigned worker = 0;
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);

	struct starpu_sched_ctx_iterator it;

	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
	{
		worker = workers->get_next(workers, &it);
		starpu_pthread_mutex_t *sched_mutex;
		starpu_pthread_cond_t *sched_cond;
		starpu_worker_get_sched_condition(worker, &sched_mutex, &sched_cond);
		starpu_pthread_mutex_lock(sched_mutex);
		starpu_pthread_cond_signal(sched_cond);
		starpu_pthread_mutex_unlock(sched_mutex);
	}
#endif
	return 0;
}

static struct starpu_task *pop_task_from_ready_queue(unsigned sched_ctx_id)
{
	const unsigned workerid = starpu_worker_get_id();
	struct _starpu_hpbasic_ready_queue *data = (struct _starpu_hpbasic_ready_queue*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	struct starpu_task *task = NULL;
	unsigned queueId;

	starpu_pthread_mutex_lock(&data->hp_mutex);
	if(starpu_worker_get_type(workerid) == STARPU_CPU_WORKER)
	{
		/* CPU worker selects ready queue in order of decreasing acceleration factor (CPU time /GPU time) */
		for(queueId=0; queueId<data->types_of_tasks && task == NULL; queueId++)
			task = starpu_st_fifo_taskq_pop_local_task(data->task_queues[queueId].tasks_queue);
	}
	else if(starpu_worker_get_type(workerid) == STARPU_CUDA_WORKER)
	{
		/* CPU worker selects ready queue in order of decreasing acceleration factor (CPU time /GPU time) */
		for(queueId=data->types_of_tasks - 1; queueId>0 && task == NULL; queueId--)
			task = starpu_st_fifo_taskq_pop_local_task(data->task_queues[queueId].tasks_queue);
	}
	else
	{
		_STARPU_DEBUG("Platform also has some another type of worker other than CPUs and GPUs\n");
	}
	starpu_pthread_mutex_unlock(&data->hp_mutex);
	return task;
}

struct starpu_sched_policy _starpu_sched_hp_policy =
{
	.init_sched = init_hp_sched,
	.deinit_sched = deinit_hp_sched,
	.push_task = push_task_in_to_ready_queue,
	.pop_task = pop_task_from_ready_queue,
	.policy_name = "hp",
	.policy_description = "basic heteroprio strategy",
	.worker_type = STARPU_WORKER_LIST,
};
