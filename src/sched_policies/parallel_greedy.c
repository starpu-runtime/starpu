/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011  Université de Bordeaux 1
 * Copyright (C) 2011  Télécom-SudParis
 * Copyright (C) 2011  INRIA
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

#include <core/workers.h>
#include <sched_policies/fifo_queues.h>
#include <common/barrier.h>
#include <sched_policies/detect_combined_workers.h>

typedef struct pgreedy_data {
	struct _starpu_fifo_taskq *fifo;
	struct _starpu_fifo_taskq *local_fifo[STARPU_NMAXWORKERS];

	int master_id[STARPU_NMAXWORKERS];

	pthread_cond_t sched_cond;
	pthread_mutex_t sched_mutex;
} pgreedy_data;

/* XXX instead of 10, we should use some "MAX combination .."*/
static int possible_combinations_cnt[STARPU_NMAXWORKERS];
static int possible_combinations[STARPU_NMAXWORKERS][10];
static int possible_combinations_size[STARPU_NMAXWORKERS][10];


/*!!!!!!! It doesn't work with several contexts because the combined workers are constructed
  from the workers available to the program, and not to the context !!!!!!!!!!!!!!!!!!!!!!!
 */

static void pgreedy_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	struct pgreedy_data *data = (struct pgreedy_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	struct starpu_machine_topology *topology = &config->topology;

	_starpu_sched_find_worker_combinations(topology);

	unsigned workerid, i;
	unsigned ncombinedworkers;
	
	ncombinedworkers = starpu_combined_worker_get_count();

	/* Find the master of each worker. We first assign the worker as its
	 * own master, and then iterate over the different worker combinations
	 * to find the biggest combination containing this worker. */
	for(i = 0; i < nworkers; i++)
	{
		workerid = workerids[i];
		
		int cnt = possible_combinations_cnt[workerid]++;
		possible_combinations[workerid][cnt] = workerid;
		possible_combinations_size[workerid][cnt] = 1;
		
		data->master_id[workerid] = workerid;
	}
	
	
	for (i = 0; i < ncombinedworkers; i++)
	{
		int workerid = nworkers + i;

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

		unsigned master = data->master_id[workerid];

		/* All masters use the same condition/mutex */
		if (workerid == master)
			starpu_worker_set_sched_condition(sched_ctx_id, workerid, &data->sched_mutex, &data->sched_cond);
		else
			starpu_worker_init_sched_condition(sched_ctx_id, workerid);
	}

#if 0
	for(i = 0; i < nworkers; i++)
        {
		workerid = workerids[i];

		fprintf(stderr, "MASTER of %d = %d\n", workerid, master_id[workerid]);
	}
#endif

}

static void pgreedy_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	struct pgreedy_data *data = (struct pgreedy_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);
	int workerid;
	unsigned i;
	for(i = 0; i < nworkers; i++)
        {
		workerid = workerids[i];
		_starpu_destroy_fifo(data->local_fifo[workerid]);
		unsigned master = data->master_id[workerid];
		if(workerid != master)
			starpu_worker_deinit_sched_condition(sched_ctx_id, workerid);
		else
			starpu_worker_set_sched_condition(sched_ctx_id, workerid, NULL, NULL);
	}
}

static void initialize_pgreedy_policy(unsigned sched_ctx_id) 
{
	starpu_create_worker_collection_for_sched_ctx(sched_ctx_id, WORKER_LIST);

	struct pgreedy_data *data = (struct pgreedy_data*)malloc(sizeof(pgreedy_data));
	/* masters pick tasks from that queue */
	data->fifo = _starpu_create_fifo();

	_STARPU_PTHREAD_MUTEX_INIT(&data->sched_mutex, NULL);
	_STARPU_PTHREAD_COND_INIT(&data->sched_cond, NULL);

	starpu_set_sched_ctx_policy_data(sched_ctx_id, (void*)data);
}

static void deinitialize_pgreedy_policy(unsigned sched_ctx_id) 
{
	/* TODO check that there is no task left in the queue */
	struct pgreedy_data *data = (struct pgreedy_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);

	/* deallocate the job queue */
	_starpu_destroy_fifo(data->fifo);

	_STARPU_PTHREAD_MUTEX_DESTROY(&data->sched_mutex);
	_STARPU_PTHREAD_COND_DESTROY(&data->sched_cond);

	starpu_delete_worker_collection_for_sched_ctx(sched_ctx_id);

	free(data);	
}

static int push_task_pgreedy_policy(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	pthread_mutex_t *changing_ctx_mutex = starpu_get_changing_ctx_mutex(sched_ctx_id);
	unsigned nworkers;
        int ret_val = -1;

	/* if the context has no workers return */
        _STARPU_PTHREAD_MUTEX_LOCK(changing_ctx_mutex);
        nworkers = starpu_get_nworkers_of_sched_ctx(sched_ctx_id);
        if(nworkers == 0)
        {
                _STARPU_PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);
                return ret_val;
        }
	struct pgreedy_data *data = (struct pgreedy_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);

	ret_val = _starpu_fifo_push_task(data->fifo, &data->sched_mutex, &data->sched_cond, task);
	_STARPU_PTHREAD_MUTEX_UNLOCK(changing_ctx_mutex);

	return ret_val;
}

static struct starpu_task *pop_task_pgreedy_policy(unsigned sched_ctx_id)
{
	struct pgreedy_data *data = (struct pgreedy_data*)starpu_get_sched_ctx_policy_data(sched_ctx_id);

	int workerid = starpu_worker_get_id();

	/* If this is not a CPU, then the worker simply grabs tasks from the fifo */
	if (starpu_worker_get_type(workerid) != STARPU_CPU_WORKER)
		return  _starpu_fifo_pop_task(data->fifo, workerid);

	int master = data->master_id[workerid];

	if (workerid == master)
	{
		/* The worker is a master */
		struct starpu_task *task = _starpu_fifo_pop_task(data->fifo, workerid);

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
			/* The master needs to dispatch the task between the
			 * different combined workers */
			struct _starpu_combined_worker *combined_worker;
			combined_worker = _starpu_get_combined_worker_struct(best_workerid);
			int worker_size = combined_worker->worker_size;
			int *combined_workerid = combined_worker->combined_workerid;

			struct _starpu_job *j = _starpu_get_job_associated_to_task(task);
			j->task_size = worker_size;
			j->combined_workerid = best_workerid;
			j->active_task_alias_count = 0;

			//fprintf(stderr, "POP -> size %d best_size %d\n", worker_size, best_size);

			_STARPU_PTHREAD_BARRIER_INIT(&j->before_work_barrier, NULL, worker_size);
			_STARPU_PTHREAD_BARRIER_INIT(&j->after_work_barrier, NULL, worker_size);

			/* Dispatch task aliases to the different slaves */
			for (i = 1; i < worker_size; i++)
			{
				struct starpu_task *alias = _starpu_create_task_alias(task);
				int local_worker = combined_workerid[i];

				pthread_mutex_t *sched_mutex;
				pthread_cond_t *sched_cond;
				starpu_worker_get_sched_condition(sched_ctx_id, master, &sched_mutex, &sched_cond);
				_starpu_fifo_push_task(data->local_fifo[local_worker],
						       sched_mutex, sched_cond, alias);
			}

			/* The master also manipulated an alias */
			struct starpu_task *master_alias = _starpu_create_task_alias(task);
			return master_alias;
		}
	}
	else
	{
		/* The worker is a slave */
		return _starpu_fifo_pop_task(data->local_fifo[workerid], workerid);
	}
}

struct starpu_sched_policy _starpu_sched_pgreedy_policy =
{
	.init_sched = initialize_pgreedy_policy,
	.deinit_sched = deinitialize_pgreedy_policy,
	.add_workers = pgreedy_add_workers,
	.remove_workers = pgreedy_remove_workers,
	.push_task = push_task_pgreedy_policy,
	.pop_task = pop_task_pgreedy_policy,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "pgreedy",
	.policy_description = "parallel greedy policy"
};
