/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2015  Université de Bordeaux
 * Copyright (C) 2010, 2011, 2012, 2013, 2014, 2015  CNRS
 * Copyright (C) 2011, 2012  INRIA
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

/* Work stealing policy */

#include <float.h>

#include <core/workers.h>
#include <sched_policies/fifo_queues.h>
#include <core/debug.h>
#include <starpu_bitmap.h>

struct _starpu_lws_data
{
	struct _starpu_fifo_taskq **queue_array;
	int **proxlist;
	unsigned last_pop_worker;
	unsigned last_push_worker;
};


#ifdef STARPU_HAVE_HWLOC

/* Return a worker to steal a task from. The worker is selected
 * according to the proximity list built using the info on te
 * architecture provided by hwloc */
static unsigned select_victim_neighborhood(unsigned sched_ctx_id, int workerid)
{

	struct _starpu_lws_data *ws = (struct _starpu_lws_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	int nworkers = starpu_sched_ctx_get_nworkers(sched_ctx_id);

	int i;
	int neighbor;
	for(i=0; i<nworkers; i++)
	{
		neighbor = ws->proxlist[workerid][i];
		int ntasks = ws->queue_array[neighbor]->ntasks;

		if (ntasks)
			return neighbor;
	}

	return workerid;
}
#else
/* Return a worker to steal a task from. The worker is selected
 * in a round-robin fashion */
static unsigned select_victim_round_robin(unsigned sched_ctx_id)
{
	struct _starpu_lws_data *ws = (struct _starpu_lws_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned worker = ws->last_pop_worker;
	unsigned nworkers = starpu_sched_ctx_get_nworkers(sched_ctx_id);

	starpu_pthread_mutex_t *victim_sched_mutex;
	starpu_pthread_cond_t *victim_sched_cond;

	/* If the worker's queue is empty, let's try
	 * the next ones */
	while (1)
	{
		unsigned ntasks;

		starpu_worker_get_sched_condition(worker, &victim_sched_mutex, &victim_sched_cond);
		ntasks = ws->queue_array[worker]->ntasks;
		if (ntasks)
			break;

		worker = (worker + 1) % nworkers;
		if (worker == ws->last_pop_worker)
		{
			/* We got back to the first worker,
			 * don't go in infinite loop */
			break;
		}
	}

	ws->last_pop_worker = (worker + 1) % nworkers;

	return worker;
}


#endif


/**
 * Return a worker to whom add a task.
 * Selecting a worker is done in a round-robin fashion.
 */
static unsigned select_worker_round_robin(unsigned sched_ctx_id)
{
	struct _starpu_lws_data *ws = (struct _starpu_lws_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	unsigned worker = ws->last_push_worker;
	unsigned nworkers = starpu_sched_ctx_get_nworkers(sched_ctx_id);
	/* TODO: use an atomic update operation for this */
	ws->last_push_worker = (ws->last_push_worker + 1) % nworkers;

	return worker;
}


/**
 * Return a worker from which a task can be stolen.
 */
static inline unsigned select_victim(unsigned sched_ctx_id, int workerid STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_HAVE_HWLOC
	return select_victim_neighborhood(sched_ctx_id, workerid);
#else
	return select_victim_round_robin(sched_ctx_id);
#endif
}

/**
 * Return a worker on whose queue a task can be pushed. This is only
 * needed when the push is done by the master
 */
static inline unsigned select_worker(unsigned sched_ctx_id)
{
	return select_worker_round_robin(sched_ctx_id);
}


static struct starpu_task *lws_pop_task(unsigned sched_ctx_id)
{
	struct _starpu_lws_data *ws = (struct _starpu_lws_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	struct starpu_task *task = NULL;

	int workerid = starpu_worker_get_id();

	STARPU_ASSERT(workerid != -1);

	task = _starpu_fifo_pop_task(ws->queue_array[workerid], workerid);
	if (task)
	{
		/* there was a local task */
		/* printf("Own    task!%d\n",workerid); */
		return task;
	}
	starpu_pthread_mutex_t *worker_sched_mutex;
	starpu_pthread_cond_t *worker_sched_cond;
	starpu_worker_get_sched_condition(workerid, &worker_sched_mutex, &worker_sched_cond);

	/* Note: Releasing this mutex before taking the victim mutex, to avoid interlock*/
	STARPU_PTHREAD_MUTEX_UNLOCK(worker_sched_mutex);


	/* we need to steal someone's job */
	unsigned victim = select_victim(sched_ctx_id, workerid);

	starpu_pthread_mutex_t *victim_sched_mutex;
	starpu_pthread_cond_t *victim_sched_cond;

	starpu_worker_get_sched_condition(victim, &victim_sched_mutex, &victim_sched_cond);
	STARPU_PTHREAD_MUTEX_LOCK(victim_sched_mutex);

	task = _starpu_fifo_pop_task(ws->queue_array[victim], workerid);
	if (task)
	{
		_STARPU_TRACE_WORK_STEALING(workerid, victim);
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(victim_sched_mutex);

	STARPU_PTHREAD_MUTEX_LOCK(worker_sched_mutex);
	if(!task)
	{
		task = _starpu_fifo_pop_task(ws->queue_array[workerid], workerid);
		if (task)
		{
			/* there was a local task */
			return task;
		}
	}

	return task;
}

static int lws_push_task(struct starpu_task *task)
{
	unsigned sched_ctx_id = task->sched_ctx;
	struct _starpu_lws_data *ws = (struct _starpu_lws_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	int workerid = starpu_worker_get_id();

	/* If the current thread is not a worker but
	 * the main thread (-1), we find the better one to
	 * put task on its queue */
	if (workerid == -1)
		workerid = select_worker(sched_ctx_id);

	/* int workerid = starpu_worker_get_id(); */
	/* print_neighborhood(sched_ctx_id, 0); */

	starpu_pthread_mutex_t *sched_mutex;
	starpu_pthread_cond_t *sched_cond;
	starpu_worker_get_sched_condition(workerid, &sched_mutex, &sched_cond);
	STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);

	_starpu_fifo_push_task(ws->queue_array[workerid], task);

	starpu_push_task_end(task);

	STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);

#if !defined(STARPU_NON_BLOCKING_DRIVERS) || defined(STARPU_SIMGRID)
	/* TODO: implement fine-grain signaling, similar to what eager does */
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
	struct starpu_sched_ctx_iterator it;

	workers->init_iterator(workers, &it);
	while(workers->has_next(workers, &it))
		starpu_wake_worker(workers->get_next(workers, &it));
#endif



	return 0;
}

static void lws_add_workers(unsigned sched_ctx_id, int *workerids,unsigned nworkers)
{
	struct _starpu_lws_data *ws = (struct _starpu_lws_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	unsigned i;
	int workerid;

	for (i = 0; i < nworkers; i++)
	{
		workerid = workerids[i];
		starpu_sched_ctx_worker_shares_tasks_lists(workerid, sched_ctx_id);
		ws->queue_array[workerid] = _starpu_create_fifo();

		/* Tell helgrid that we are fine with getting outdated values,
		 * this is just an estimation */
		STARPU_HG_DISABLE_CHECKING(ws->queue_array[workerid]->ntasks);

		ws->queue_array[workerid]->nprocessed = 0;
		ws->queue_array[workerid]->ntasks = 0;
	}


#ifdef STARPU_HAVE_HWLOC
	/* Build a proximity list for every worker. It is cheaper to
	 * build this once and then use it for popping tasks rather
	 * than traversing the hwloc tree every time a task must be
	 * stolen */
	ws->proxlist = (int**)malloc(nworkers*sizeof(int*));
	struct starpu_worker_collection *workers = starpu_sched_ctx_get_worker_collection(sched_ctx_id);
	struct starpu_tree *tree = (struct starpu_tree*)workers->workerids;
	for (i = 0; i < nworkers; i++)
	{
		workerid = workerids[i];
		ws->proxlist[workerid] = (int*)malloc(nworkers*sizeof(int));
		int bindid;

		struct starpu_tree *neighbour = NULL;
		struct starpu_sched_ctx_iterator it;

		workers->init_iterator(workers, &it);

		bindid   = starpu_worker_get_bindid(workerid);
		it.value = starpu_tree_get(tree, bindid);
		int cnt = 0;
		for(;;)
		{
			neighbour = (struct starpu_tree*)it.value;
			int neigh_workerids[STARPU_NMAXWORKERS];
			int neigh_nworkers = _starpu_worker_get_workerids(neighbour->id, neigh_workerids);
			int w;
			for(w = 0; w < neigh_nworkers; w++)
			{
				if(!it.visited[neigh_workerids[w]] && workers->present[neigh_workerids[w]])
				{
					ws->proxlist[workerid][cnt++] = neigh_workerids[w];
					it.visited[neigh_workerids[w]] = 1;
				}
			}
			if(!workers->has_next(workers, &it))
				break;
			it.value = it.possible_value;
			it.possible_value = NULL;
		}
	}
#endif
}

static void lws_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	struct _starpu_lws_data *ws = (struct _starpu_lws_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	unsigned i;
	int workerid;

	for (i = 0; i < nworkers; i++)
	{
		workerid = workerids[i];
		_starpu_destroy_fifo(ws->queue_array[workerid]);
#ifdef STARPU_HAVE_HWLOC
		free(ws->proxlist[workerid]);
#endif
	}
}

static void lws_initialize_policy(unsigned sched_ctx_id)
{
#ifdef STARPU_HAVE_HWLOC
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_TREE);
#else
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
#endif

	struct _starpu_lws_data *ws = (struct _starpu_lws_data*)malloc(sizeof(struct _starpu_lws_data));
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)ws);

	ws->last_pop_worker = 0;
	ws->last_push_worker = 0;

	/* unsigned nw = starpu_sched_ctx_get_nworkers(sched_ctx_id); */
	unsigned nw = starpu_worker_get_count();
	ws->queue_array = (struct _starpu_fifo_taskq**)malloc(nw*sizeof(struct _starpu_fifo_taskq*));

}

static void lws_deinit_policy(unsigned sched_ctx_id)
{
	struct _starpu_lws_data *ws = (struct _starpu_lws_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	free(ws->queue_array);
#ifdef STARPU_HAVE_HWLOC
	free(ws->proxlist);
#endif
	free(ws);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}

struct starpu_sched_policy _starpu_sched_lws_policy =
{
	.init_sched = lws_initialize_policy,
	.deinit_sched = lws_deinit_policy,
	.add_workers = lws_add_workers,
	.remove_workers = lws_remove_workers,
	.push_task = lws_push_task,
	.pop_task = lws_pop_task,
	.pre_exec_hook = NULL,
	.post_exec_hook = NULL,
	.pop_every_task = NULL,
	.policy_name = "lws",
	.policy_description = "locality work stealing"
};
