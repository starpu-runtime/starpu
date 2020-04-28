/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 *	This is just a test policy for using task graph information
 *
 *	We keep tasks in the fifo queue, and store the graph of tasks, until we
 *	get the do_schedule call from the application, which tells us all tasks
 *	were queued, and we can now compute task depths or descendants and let a simple
 *	central-queue greedy algorithm proceed.
 *
 *	TODO: let workers starting running tasks before the whole graph is submitted?
 */

#include <starpu_scheduler.h>
#include <sched_policies/fifo_queues.h>
#include <sched_policies/prio_deque.h>
#include <common/graph.h>
#include <common/thread.h>
#include <starpu_bitmap.h>
#include <core/task.h>
#include <core/workers.h>

struct _starpu_graph_test_policy_data
{
	struct _starpu_fifo_taskq *fifo;	/* Bag of tasks which are ready before do_schedule is called */
	struct _starpu_prio_deque prio_cpu;
	struct _starpu_prio_deque prio_gpu;
	starpu_pthread_mutex_t policy_mutex;
	struct starpu_bitmap *waiters;
	unsigned computed;
	unsigned descendants;			/* Whether we use descendants, or depths, for priorities */
};

static void initialize_graph_test_policy(unsigned sched_ctx_id)
{
	struct _starpu_graph_test_policy_data *data;
	_STARPU_MALLOC(data, sizeof(struct _starpu_graph_test_policy_data));

	/* there is only a single queue in that trivial design */
	data->fifo =  _starpu_create_fifo();
	 _starpu_prio_deque_init(&data->prio_cpu);
	 _starpu_prio_deque_init(&data->prio_gpu);
	data->waiters = starpu_bitmap_create();
	data->computed = 0;
	data->descendants = starpu_get_env_number_default("STARPU_SCHED_GRAPH_TEST_DESCENDANTS", 0);

	_starpu_graph_record = 1;

	 /* Tell helgrind that it's fine to check for empty fifo in
	  * pop_task_graph_test_policy without actual mutex (it's just an integer)
	  */
	STARPU_HG_DISABLE_CHECKING(data->fifo->ntasks);

	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)data);
	STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
}

static void deinitialize_graph_test_policy(unsigned sched_ctx_id)
{
	struct _starpu_graph_test_policy_data *data = (struct _starpu_graph_test_policy_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	struct _starpu_fifo_taskq *fifo = data->fifo;

	STARPU_ASSERT(starpu_task_list_empty(&fifo->taskq));

	/* deallocate the job queue */
	_starpu_destroy_fifo(fifo);
	 _starpu_prio_deque_destroy(&data->prio_cpu);
	 _starpu_prio_deque_destroy(&data->prio_gpu);
	starpu_bitmap_destroy(data->waiters);

	_starpu_graph_record = 0;
	STARPU_PTHREAD_MUTEX_DESTROY(&data->policy_mutex);
	free(data);
}

/* Push the given task on CPU or GPU prio list, using a dumb heuristic */
static struct _starpu_prio_deque *select_prio(unsigned sched_ctx_id, struct _starpu_graph_test_policy_data *data, struct starpu_task *task)
{
	int cpu_can = 0, gpu_can = 0;
	double cpu_speed = 0.;
	double gpu_speed = 0.;

	/* Compute how fast CPUs can compute it, and how fast GPUs can compute it */
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
	struct starpu_sched_ctx_iterator it;
	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
	{
		unsigned worker = workers->get_next(workers, &it);
		if (!starpu_worker_can_execute_task(worker, task, 0))
			/* This worker can not execute this task, don't count it */
			continue;

		if (starpu_worker_get_type(worker) == STARPU_CPU_WORKER)
			/* At least one CPU can run it */
			cpu_can = 1;
		else
			/* At least one GPU can run it */
			gpu_can = 1;

		/* Get expected task duration for this worker */
		struct starpu_perfmodel_arch* perf_arch = starpu_worker_get_perf_archtype(worker, sched_ctx_id);
		double length = starpu_task_expected_length(task, perf_arch, 0);
		double power;

		if (isnan(length))
			/* We don't have an estimation yet */
			length = 0.;
		if (length == 0.)
		{
			if (!task->cl || task->cl->model == NULL)
			{
				static unsigned _warned;
				if (STARPU_ATOMIC_ADD(&_warned, 1) == 1)
				{
					_STARPU_DISP("Warning: graph_test needs performance models for all tasks, including %s\n",
							starpu_task_get_name(task));
				}
				else
				{
					(void)STARPU_ATOMIC_ADD(&_warned, -1);
				}
			}
			power = 0.;
		}
		else
			power = 1./length;

		/* Add the computation power to the CPU or GPU pool */
		if (starpu_worker_get_type(worker) == STARPU_CPU_WORKER)
			cpu_speed += power;
		else
			gpu_speed += power;
	}

	/* Decide to push on CPUs or GPUs depending on the overall computation power */
	if (!gpu_can || (cpu_can && cpu_speed > gpu_speed))
		return &data->prio_cpu;
	else
		return &data->prio_gpu;

}

static void set_priority(void *_data, struct _starpu_graph_node *node)
{
	struct _starpu_graph_test_policy_data *data = _data;
	starpu_worker_relax_on();
	STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
	starpu_worker_relax_off();
	struct _starpu_job *job = node->job;
	if (job)
	{
		if (data->descendants)
			job->task->priority = node->descendants;
		else
			job->task->priority = node->depth;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);
}

static void do_schedule_graph_test_policy(unsigned sched_ctx_id)
{
	struct _starpu_graph_test_policy_data *data = (struct _starpu_graph_test_policy_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	starpu_worker_relax_on();
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	starpu_worker_relax_off();
	if (data->descendants)
		_starpu_graph_compute_descendants();
	else
		_starpu_graph_compute_depths();
	if (data->computed == 0)
	{
		data->computed = 1;

		/* FIXME: if data->computed already == 1, some tasks may already have been pushed to priority stage '0' in
		 * push_task_graph_test_policy, then if we change the priority here, the stage lookup to remove the task
		 * will get the wrong stage */
		_starpu_graph_foreach(set_priority, data);
	}

	/* Now that we have priorities, move tasks from bag to priority queue */
	while(!_starpu_fifo_empty(data->fifo))
	{
		struct starpu_task *task = _starpu_fifo_pop_task(data->fifo, -1);
		struct _starpu_prio_deque *prio = select_prio(sched_ctx_id, data, task);
		_starpu_prio_deque_push_back_task(prio, task);
	}

	/* And unleash the beast! */
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
	struct starpu_sched_ctx_iterator it;
#ifdef STARPU_NON_BLOCKING_DRIVERS
	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
	{
		/* Tell each worker is shouldn't sleep any more */
		unsigned worker = workers->get_next(workers, &it);
		starpu_bitmap_unset(data->waiters, worker);
	}
#endif
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);

#if !defined(STARPU_NON_BLOCKING_DRIVERS) || defined(STARPU_SIMGRID)
	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
	{
		/* Wake each worker */
		unsigned worker = workers->get_next(workers, &it);
		starpu_wake_worker_relax_light(worker);
	}
#endif
}

static int push_task_graph_test_policy(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	struct _starpu_graph_test_policy_data *data = (struct _starpu_graph_test_policy_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	starpu_worker_relax_on();
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	starpu_worker_relax_off();
	if (!data->computed)
	{
		/* Priorities are not computed, leave the task in the bag for now */
		starpu_task_list_push_back(&data->fifo->taskq,task);
		data->fifo->ntasks++;
		data->fifo->nprocessed++;
		starpu_push_task_end(task);
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		return 0;
	}

	/* Priorities are computed, we can push to execution */
	struct _starpu_prio_deque *prio = select_prio(sched_ctx_id, data, task);
	_starpu_prio_deque_push_back_task(prio, task);

	starpu_push_task_end(task);

	/*if there are no tasks block */
	/* wake people waiting for a task */
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);

	struct starpu_sched_ctx_iterator it;
#ifndef STARPU_NON_BLOCKING_DRIVERS
	char dowake[STARPU_NMAXWORKERS] = { 0 };
#endif

	workers->init_iterator_for_parallel_tasks(workers, &it, task);
	while(workers->has_next(workers, &it))
	{
		unsigned worker = workers->get_next(workers, &it);

#ifdef STARPU_NON_BLOCKING_DRIVERS
		if (!starpu_bitmap_get(data->waiters, worker))
			/* This worker is not waiting for a task */
			continue;
#endif
		if (prio == &data->prio_cpu && starpu_worker_get_type(worker) != STARPU_CPU_WORKER)
			/* This worker doesn't pop from the queue we have filled */
			continue;
		if (prio == &data->prio_gpu && starpu_worker_get_type(worker) == STARPU_CPU_WORKER)
			/* This worker doesn't pop from the queue we have filled */
			continue;

		if (starpu_worker_can_execute_task_first_impl(worker, task, NULL))
		{
			/* It can execute this one, tell him! */
#ifdef STARPU_NON_BLOCKING_DRIVERS
			starpu_bitmap_unset(data->waiters, worker);
			/* We really woke at least somebody, no need to wake somebody else */
			break;
#else
			dowake[worker] = 1;
#endif
		}
	}
	/* Let the task free */
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);

#if !defined(STARPU_NON_BLOCKING_DRIVERS) || defined(STARPU_SIMGRID)
	/* Now that we have a list of potential workers, try to wake one */

	workers->init_iterator_for_parallel_tasks(workers, &it, task);
	while(workers->has_next(workers, &it))
	{
		unsigned worker = workers->get_next(workers, &it);
		if (dowake[worker])
		{
			if (starpu_wake_worker_relax_light(worker))
				break; // wake up a single worker
		}
	}
#endif

	return 0;
}

static struct starpu_task *pop_task_graph_test_policy(unsigned sched_ctx_id)
{
	struct starpu_task *chosen_task = NULL;
	unsigned workerid = starpu_worker_get_id_check();
	struct _starpu_graph_test_policy_data *data = (struct _starpu_graph_test_policy_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	struct _starpu_prio_deque *prio;

	if (starpu_worker_get_type(workerid) == STARPU_CPU_WORKER)
		prio = &data->prio_cpu;
	else
		prio = &data->prio_gpu;

	/* block until some event happens */
	/* Here helgrind would shout that this is unprotected, this is just an
	 * integer access, and we hold the sched mutex, so we can not miss any
	 * wake up. */
	if (!STARPU_RUNNING_ON_VALGRIND && _starpu_prio_deque_is_empty(prio))
		return NULL;

#ifdef STARPU_NON_BLOCKING_DRIVERS
	if (!STARPU_RUNNING_ON_VALGRIND && !data->computed)
		/* Not computed yet */
		return NULL;
	if (!STARPU_RUNNING_ON_VALGRIND && starpu_bitmap_get(data->waiters, workerid))
		/* Nobody woke us, avoid bothering the mutex */
		return NULL;
#endif

	starpu_worker_relax_on();
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	starpu_worker_relax_off();
	if (!data->computed)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		return NULL;
	}

	chosen_task = _starpu_prio_deque_pop_task_for_worker(prio, workerid, NULL);
	if (!chosen_task)
		/* Tell pushers that we are waiting for tasks for us */
		starpu_bitmap_set(data->waiters, workerid);

	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);

	return chosen_task;
}

struct starpu_sched_policy _starpu_sched_graph_test_policy =
{
	.init_sched = initialize_graph_test_policy,
	.deinit_sched = deinitialize_graph_test_policy,
	.do_schedule = do_schedule_graph_test_policy,
	.push_task = push_task_graph_test_policy,
	.pop_task = pop_task_graph_test_policy,
	.policy_name = "graph_test",
	.policy_description = "test policy for using graphs in scheduling decisions",
	.worker_type = STARPU_WORKER_LIST,
};
