/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2013  Université de Bordeaux 1
 * Copyright (C) 2011  Télécom-SudParis
 * Copyright (C) 2011-2013  INRIA
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

struct _starpu_peager_data
{
	struct _starpu_fifo_taskq *fifo;
	struct _starpu_fifo_taskq *local_fifo[STARPU_NMAXWORKERS];

	int master_id[STARPU_NMAXWORKERS];
        starpu_pthread_mutex_t policy_mutex;
};

#define STARPU_NMAXCOMBINED_WORKERS 520 
/* instead of STARPU_NMAXCOMBINED_WORKERS, we should use some "MAX combination .."*/
static int possible_combinations_cnt[STARPU_NMAXWORKERS];
static int possible_combinations[STARPU_NMAXWORKERS][STARPU_NMAXCOMBINED_WORKERS];
static int possible_combinations_size[STARPU_NMAXWORKERS][STARPU_NMAXCOMBINED_WORKERS];


/*!!!!!!! It doesn't work with several contexts because the combined workers are constructed
  from the workers available to the program, and not to the context !!!!!!!!!!!!!!!!!!!!!!!
 */

static void peager_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	_starpu_sched_find_worker_combinations(workerids, nworkers);
	struct _starpu_peager_data *data = (struct _starpu_peager_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned nbasic_workers = starpu_worker_get_count();
	unsigned ncombined_workers= starpu_combined_worker_get_count();
	unsigned workerid, i;

	/* Find the master of each worker. We first assign the worker as its
	 * own master, and then iterate over the different worker combinations
	 * to find the biggest combination containing this worker. */
	for(i = 0; i < nworkers; i++)
	{
		workerid = workerids[i];
		starpu_sched_ctx_worker_shares_tasks_lists(workerid, sched_ctx_id);
		int cnt = possible_combinations_cnt[workerid]++;
		possible_combinations[workerid][cnt] = workerid;
		possible_combinations_size[workerid][cnt] = 1;

		data->master_id[workerid] = workerid;
	}


	for (i = 0; i < ncombined_workers; i++)
	{
		workerid = nbasic_workers + i;

		/* Note that we ASSUME that the workers are sorted by size ! */
		int *workers;
		int size;
		starpu_combined_worker_get_description(workerid, &size, &workers);

		int master = workers[0];
		int j;
		for (j = 0; j < size; j++)
		{
			if (data->master_id[workers[j]] > master)
				data->master_id[workers[j]] = master;
			int cnt = possible_combinations_cnt[workers[j]]++;
			possible_combinations[workers[j]][cnt] = workerid;
			possible_combinations_size[workers[j]][cnt] = size;
		}
	}

	for(i = 0; i < nworkers; i++)
	{
		workerid = workerids[i];
		
		/* slaves pick up tasks from their local queue, their master
		 * will put tasks directly in that local list when a parallel
		 * tasks comes. */
		data->local_fifo[workerid] = _starpu_create_fifo();
	}
	
#if 0
	for(i = 0; i < nworkers; i++)
        {
		workerid = workerids[i];

		fprintf(stderr, "MASTER of %d = %d\n", workerid, master_id[workerid]);
	}
#endif
}

static void peager_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	struct _starpu_peager_data *data = (struct _starpu_peager_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	int workerid;
	unsigned i;
	for(i = 0; i < nworkers; i++)
        {
		workerid = workerids[i];
		if(!starpu_worker_is_combined_worker(workerid))
			_starpu_destroy_fifo(data->local_fifo[workerid]);
	}
}

static void initialize_peager_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);

	struct _starpu_peager_data *data = (struct _starpu_peager_data*)malloc(sizeof(struct _starpu_peager_data));
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

	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
        STARPU_PTHREAD_MUTEX_DESTROY(&data->policy_mutex);

	free(data);
}

static int push_task_peager_policy(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	int ret_val = -1;
	
	struct _starpu_peager_data *data = (struct _starpu_peager_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	
	STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
	ret_val = _starpu_fifo_push_task(data->fifo, task);
	starpu_push_task_end(task);
	STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);

        /*if there are no tasks block */
        /* wake people waiting for a task */
        int worker = -1;
        struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);

        struct starpu_sched_ctx_iterator it;
        if(workers->init_iterator)
                workers->init_iterator(workers, &it);


	while(workers->has_next(workers, &it))
	{
		worker = workers->get_next(workers, &it);
		int master = data->master_id[worker];
		/* If this is not a CPU or a MIC, then the worker simply grabs tasks from the fifo */
		if ((!starpu_worker_is_combined_worker(worker) && 
		    starpu_worker_get_type(worker) != STARPU_MIC_WORKER &&
		    starpu_worker_get_type(worker) != STARPU_CPU_WORKER)  
			|| (master == worker))
		{
			starpu_pthread_mutex_t *sched_mutex;
			starpu_pthread_cond_t *sched_cond;
			starpu_worker_get_sched_condition(worker, &sched_mutex, &sched_cond);
			STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
			STARPU_PTHREAD_COND_SIGNAL(sched_cond);
			STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);
		}
	}

	return ret_val;
}

static struct starpu_task *pop_task_peager_policy(unsigned sched_ctx_id)
{
	struct _starpu_peager_data *data = (struct _starpu_peager_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	int workerid = starpu_worker_get_id();

	/* If this is not a CPU or a MIC, then the worker simply grabs tasks from the fifo */
	if (starpu_worker_get_type(workerid) != STARPU_CPU_WORKER && starpu_worker_get_type(workerid) != STARPU_MIC_WORKER)
	{
		struct starpu_task *task = NULL;
		STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
		task = _starpu_fifo_pop_task(data->fifo, workerid);
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);

		return task;
	}

	int master = data->master_id[workerid];

	//_STARPU_DEBUG("workerid:%d, master:%d\n",workerid,master);


	if (master == workerid)
	{
		/* The worker is a master */
		struct starpu_task *task = NULL;
		STARPU_PTHREAD_MUTEX_LOCK(&data->policy_mutex);
		task = _starpu_fifo_pop_task(data->fifo, workerid);
		STARPU_PTHREAD_MUTEX_UNLOCK(&data->policy_mutex);

		if (!task)
			return NULL;

		/* Find the largest compatible worker combination */
		int best_size = -1;
		int best_workerid = -1;
		int i;
		for (i = 0; i < possible_combinations_cnt[master]; i++)
		{
			if (possible_combinations_size[workerid][i] > best_size)
			{
				int combined_worker = possible_combinations[workerid][i];
				if (starpu_combined_worker_can_execute_task(combined_worker, task, 0))
				{
					best_size = possible_combinations_size[workerid][i];
					best_workerid = combined_worker;
				}
			}
		}

		/* In case nobody can execute this task, we let the master
		 * worker take it anyway, so that it can discard it afterward.
		 * */
		if (best_workerid == -1)
			return task;

		/* Is this a basic worker or a combined worker ? */
		int nbasic_workers = (int)starpu_worker_get_count();
		int is_basic_worker = (best_workerid < nbasic_workers);
		if (is_basic_worker)
		{

			/* The master is alone */
			return task;
		}
		else
		{
			starpu_parallel_task_barrier_init(task, best_workerid);
			int worker_size = 0;
			int *combined_workerid;
			starpu_combined_worker_get_description(best_workerid, &worker_size, &combined_workerid);

			/* Dispatch task aliases to the different slaves */
			for (i = 1; i < worker_size; i++)
			{
				struct starpu_task *alias = starpu_task_dup(task);
				int local_worker = combined_workerid[i];

				starpu_pthread_mutex_t *sched_mutex;
				starpu_pthread_cond_t *sched_cond;
				starpu_worker_get_sched_condition(local_worker, &sched_mutex, &sched_cond);

				STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);

				_starpu_fifo_push_task(data->local_fifo[local_worker], alias);

				STARPU_PTHREAD_COND_SIGNAL(sched_cond);
				STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);

			}

			/* The master also manipulated an alias */
			struct starpu_task *master_alias = starpu_task_dup(task);
			return master_alias;
		}
	}
	else
	{
		/* The worker is a slave */
		return _starpu_fifo_pop_task(data->local_fifo[workerid], workerid);
	}
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
	.policy_description = "parallel eager policy"
};
