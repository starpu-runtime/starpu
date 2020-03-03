/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
 * Copyright (C) 2013       Thibaut Lambert
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
#include <sched_policies/fifo_queues.h>
#include <core/detect_combined_workers.h>
#include <starpu_scheduler.h>
#include <core/workers.h>

struct _starpu_peager_common_data
{
	int possible_combinations_cnt[STARPU_NMAXWORKERS];
	int *possible_combinations[STARPU_NMAXWORKERS];
	int *possible_combinations_size[STARPU_NMAXWORKERS];
	int max_combination_size[STARPU_NMAXWORKERS];
	int no_combined_workers;
	int ref_count;
};

struct _starpu_peager_common_data *_peager_common_data = NULL;

struct _starpu_peager_data
{
	starpu_pthread_mutex_t policy_mutex;
	struct _starpu_fifo_taskq *fifo;
	struct _starpu_fifo_taskq *local_fifo[STARPU_NMAXWORKERS];
};

static void initialize_peager_common(void)
{
	if (_peager_common_data == NULL)
	{
		struct _starpu_peager_common_data *common_data = NULL;
		_STARPU_CALLOC(common_data, 1, sizeof(struct _starpu_peager_common_data));
		common_data->ref_count = 1;
		_peager_common_data = common_data;

		const unsigned nbasic_workers = starpu_worker_get_count();
		unsigned i;

		starpu_sched_find_all_worker_combinations();
		const unsigned ncombined_workers = starpu_combined_worker_get_count();
		common_data->no_combined_workers = ncombined_workers == 0;

		for(i = 0; i < nbasic_workers; i++)
		{
			common_data->possible_combinations_cnt[i] = 0;
			int cnt = common_data->possible_combinations_cnt[i]++;
			/* Allocate ncombined_workers + 1 for the singleton worker itself */
			_STARPU_CALLOC(common_data->possible_combinations[i], 1+ncombined_workers, sizeof(int));
			_STARPU_CALLOC(common_data->possible_combinations_size[i], 1+ncombined_workers, sizeof(int));
			common_data->possible_combinations[i][cnt] = i;
			common_data->possible_combinations_size[i][cnt] = 1;
			common_data->max_combination_size[i] = 1;
		}

		for (i = 0; i < ncombined_workers; i++)
		{
			unsigned combined_workerid = nbasic_workers + i;
			int *workers;
			int size;
			starpu_combined_worker_get_description(combined_workerid, &size, &workers);
			int master = workers[0];
			if (size > common_data->max_combination_size[master])
			{
				common_data->max_combination_size[master] = size;
			}
			int cnt = common_data->possible_combinations_cnt[master]++;
			common_data->possible_combinations[master][cnt] = combined_workerid;
			common_data->possible_combinations_size[master][cnt] = size;
		}
	}
	else
	{
		_peager_common_data->ref_count++;
	}
}

static void deinitialize_peager_common(void)
{
	STARPU_ASSERT(_peager_common_data != NULL);
	_peager_common_data->ref_count--;
	if (_peager_common_data->ref_count == 0)
	{
		const unsigned nbasic_workers = starpu_worker_get_count();
		unsigned i;
		for(i = 0; i < nbasic_workers; i++)
		{
			free(_peager_common_data->possible_combinations[i]);
			_peager_common_data->possible_combinations[i] = NULL;
			free(_peager_common_data->possible_combinations_size[i]);
			_peager_common_data->possible_combinations_size[i] = NULL;
		}
		free(_peager_common_data);
		_peager_common_data = NULL;
	}

}

static void peager_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	if (sched_ctx_id == 0)
	{
		/* FIXME Fix scheduling contexts initialization or combined
		 * worker management, to make the initialize_peager_common()
		 * call to work right from initialize_peager_policy. For now,
		 * this fails because it causes combined workers to be generated
		 * too early. */
		initialize_peager_common();
	}
	struct _starpu_peager_data *data = (struct _starpu_peager_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned i;

	for(i = 0; i < nworkers; i++)
	{
		unsigned workerid = workerids[i];
		if(starpu_worker_is_combined_worker(workerid))
		{
			continue;
		}
		starpu_sched_ctx_worker_shares_tasks_lists(workerid, sched_ctx_id);

		/* slaves pick up tasks from their local queue, their master
		 * will put tasks directly in that local list when a parallel
		 * tasks comes. */
		data->local_fifo[workerid] = _starpu_create_fifo();
	}
}

static void peager_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	struct _starpu_peager_data *data = (struct _starpu_peager_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned i;
	for(i = 0; i < nworkers; i++)
        {
		int workerid = workerids[i];
		if(!starpu_worker_is_combined_worker(workerid))
		{
			_starpu_destroy_fifo(data->local_fifo[workerid]);
		}
	}
	if (sched_ctx_id == 0)
	{
		deinitialize_peager_common();
	}
}

static void initialize_peager_policy(unsigned sched_ctx_id)
{
	struct _starpu_peager_data *data;
	_STARPU_CALLOC(data, 1, sizeof(struct _starpu_peager_data));

	_STARPU_DISP("Warning: the peager scheduler is mostly a proof of concept and not really very optimized\n");

	/* masters pick tasks from that queue */
	data->fifo = _starpu_create_fifo();

	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)data);
        STARPU_PTHREAD_MUTEX_INIT(&data->policy_mutex, NULL);
}

static void deinitialize_peager_policy(unsigned sched_ctx_id)
{
	/* TODO check that there is no task left in the queue */
	struct _starpu_peager_data *data = (struct _starpu_peager_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	/* deallocate the job queue */
	_starpu_destroy_fifo(data->fifo);

        STARPU_PTHREAD_MUTEX_DESTROY(&data->policy_mutex);

	free(data);
}

static int push_task_peager_policy(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	int ret_val;

	struct _starpu_peager_data *data = (struct _starpu_peager_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	ret_val = _starpu_fifo_push_task(data->fifo, task);
#ifndef STARPU_NON_BLOCKING_DRIVERS
	int is_parallel_task = task->cl && task->cl->max_parallelism > 1;
#endif
	starpu_push_task_end(task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);

#ifndef STARPU_NON_BLOCKING_DRIVERS
	struct _starpu_peager_common_data *common_data = _peager_common_data;
	/* if there are no tasks block */
	/* wake people waiting for a task */
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);

	struct starpu_sched_ctx_iterator it;
	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
	{
		int workerid = workers->get_next(workers, &it);
		/* If this is not a CPU or a MIC, then the workerid simply grabs tasks from the fifo */
		if (starpu_worker_is_combined_worker(workerid))
		{
			continue;
		}
		if (starpu_worker_get_type(workerid) != STARPU_MIC_WORKER
				&& starpu_worker_get_type(workerid) != STARPU_CPU_WORKER)
		{
			starpu_wake_worker_relax_light(workerid);
			continue;
		}
		if ((!is_parallel_task) /* This is not a parallel task, can wake any workerid */
				|| (common_data->no_combined_workers) /* There is no combined workerid */
				|| (common_data->max_combination_size[workerid] > 1) /* This is a combined workerid master and the task is parallel */
		   )
		{
			starpu_wake_worker_relax_light(workerid);
		}
	}
#endif

	return ret_val;
}

static struct starpu_task *pop_task_peager_policy(unsigned sched_ctx_id)
{
	struct _starpu_peager_common_data *common_data = _peager_common_data;
	struct _starpu_peager_data *data = (struct _starpu_peager_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	int workerid = starpu_worker_get_id_check();

	/* If this is not a CPU or a MIC, then the worker simply grabs tasks from the fifo */
	if (starpu_worker_get_type(workerid) != STARPU_CPU_WORKER && starpu_worker_get_type(workerid) != STARPU_MIC_WORKER)
	{
		struct starpu_task *task;
		starpu_worker_relax_on();
		STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
		starpu_worker_relax_off();
		task = _starpu_fifo_pop_task(data->fifo, workerid);
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);

		return task;
	}

	struct starpu_task *task;
	int slave_task = 0;
	starpu_worker_relax_on();
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	starpu_worker_relax_off();
	/* check if a slave task is available in the local queue */
	task = _starpu_fifo_pop_task(data->local_fifo[workerid], workerid);
	if (!task)
	{
		/* no slave task, try to pop a task as master */
		task = _starpu_fifo_pop_task(data->fifo, workerid);
		if (task)
		{
			_STARPU_DEBUG("poping master task %p\n", task);
		}

#if 1
		/* Optional heuristic to filter out purely slave workers for parallel tasks */
		if (task && task->cl && task->cl->max_parallelism > 1 && common_data->max_combination_size[workerid] == 1 && !common_data->no_combined_workers)
		{
			/* task is potentially parallel, leave it for a combined worker master */
			_STARPU_DEBUG("pushing back master task %p\n", task);
			_starpu_fifo_push_back_task(data->fifo, task);
			task = NULL;
		}
#endif
	}
	else
	{
		slave_task = 1;
		_STARPU_DEBUG("poping slave task %p\n", task);
	}
	if (!task || slave_task)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		goto ret;
	}
	/* Find the largest compatible worker combination */
	int best_size = -1;
	int best_workerid = -1;
	int i;
	for (i = 0; i < common_data->possible_combinations_cnt[workerid]; i++)
	{
		if (common_data->possible_combinations_size[workerid][i] > best_size)
		{
			int combined_worker = common_data->possible_combinations[workerid][i];
			if (starpu_combined_worker_can_execute_task(combined_worker, task, 0))
			{
				best_size = common_data->possible_combinations_size[workerid][i];
				best_workerid = combined_worker;
			}
		}
	}
	_STARPU_DEBUG("task %p, best_workerid=%d, best_size=%d\n", task, best_workerid, best_size);

	/* In case nobody can execute this task, we let the master
	 * worker take it anyway, so that it can discard it afterward.
	 * */
	if (best_workerid == -1)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		goto ret;
	}

	/* Is this a basic worker or a combined worker ? */
	if (best_workerid < (int) starpu_worker_get_count())
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);
		/* The master is alone */
		goto ret;
	}
	starpu_parallel_task_barrier_init(task, best_workerid);
	int worker_size = 0;
	int *combined_workerid;
	starpu_combined_worker_get_description(best_workerid, &worker_size, &combined_workerid);

	_STARPU_DEBUG("dispatching task %p on combined worker %d of size %d\n", task, best_workerid, worker_size);
	/* Dispatch task aliases to the different slaves */
	for (i = 1; i < worker_size; i++)
	{
		struct starpu_task *alias = starpu_task_dup(task);
		int local_worker = combined_workerid[i];
		alias->destroy = 1;
		_STARPU_TRACE_JOB_PUSH(alias, alias->priority > 0);
		_starpu_fifo_push_task(data->local_fifo[local_worker], alias);
	}

	/* The master also manipulated an alias */
	struct starpu_task *master_alias = starpu_task_dup(task);
	master_alias->destroy = 1;
	task = master_alias;

	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);

	_STARPU_TRACE_JOB_PUSH(master_alias, master_alias->priority > 0);

	for (i = 1; i < worker_size; i++)
	{
		int local_worker = combined_workerid[i];
		starpu_worker_lock(local_worker);
#if !defined(STARPU_NON_BLOCKING_DRIVERS) || defined(STARPU_SIMGRID)
		starpu_wake_worker_locked(local_worker);
#endif
		starpu_worker_unlock(local_worker);
	}

ret:
	return task;
}

struct starpu_sched_policy _starpu_sched_peager_policy =
{
	.init_sched = initialize_peager_policy,
	.deinit_sched = deinitialize_peager_policy,
	.add_workers = peager_add_workers,
	.remove_workers = peager_remove_workers,
	.push_task = push_task_peager_policy,
	.pop_task = pop_task_peager_policy,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "peager",
	.policy_description = "parallel eager policy",
	.worker_type = STARPU_WORKER_LIST,
};
