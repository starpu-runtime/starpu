/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013,2017                                Inria
 * Copyright (C) 2013-2014,2016-2017                      CNRS
 * Copyright (C) 2014-2018                                Université de Bordeaux
 * Copyright (C) 2013                                     Simon Archipoff
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

#include <float.h>

#include <starpu_sched_component.h>
#include <starpu_scheduler.h>
#include <starpu.h>
#include <core/workers.h>
#include <core/sched_policy.h>
#include <core/task.h>

#include "prio_deque.h"

#ifdef STARPU_DEVEL
#warning TODO: locality work-stealing
#endif

struct _starpu_work_stealing_data
{
/* keep track of the work performed from the beginning of the algorithm to make
 * better decisions about which queue to child when stealing or deferring work
 */
	unsigned performed_total, last_pop_child, last_push_child;

	struct _starpu_prio_deque ** fifos;
	starpu_pthread_mutex_t ** mutexes;
	unsigned size;
};


/**
 * steal a task in a round robin way
 * return NULL if none available
 */
static struct starpu_task *  steal_task_round_robin(struct starpu_sched_component *component, int workerid)
{
	struct _starpu_work_stealing_data *wsd = component->data;
	STARPU_HG_DISABLE_CHECKING(wsd->last_pop_child);
	unsigned i = wsd->last_pop_child;
	wsd->last_pop_child = (i + 1) % component->nchildren;
	STARPU_HG_ENABLE_CHECKING(wsd->last_pop_child);
	/* If the worker's queue have no suitable tasks, let's try
	 * the next ones */
	struct starpu_task * task = NULL;
	while (1)
	{
		struct _starpu_prio_deque * fifo = wsd->fifos[i];

		STARPU_COMPONENT_MUTEX_LOCK(wsd->mutexes[i]);
		task = _starpu_prio_deque_deque_task_for_worker(fifo, workerid, NULL);
		if(task && !isnan(task->predicted))
		{
			fifo->exp_len -= task->predicted;
			fifo->nprocessed--;
		}
		STARPU_COMPONENT_MUTEX_UNLOCK(wsd->mutexes[i]);
		if(task)
		{
			starpu_sched_task_break(task);
			break;
		}

		if (i == wsd->last_pop_child)
		{
			/* We got back to the first worker,
			 * don't go in infinite loop */
			return NULL;
		}
		i = (i + 1) % component->nchildren;

	}
	return task;
}

/**
 * Return a worker to whom add a task.
 * Selecting a worker is done in a round-robin fashion.
 */
static unsigned select_worker_round_robin(struct starpu_sched_component * component)
{
	struct _starpu_work_stealing_data *ws = (struct _starpu_work_stealing_data*)component->data;
	unsigned i = (ws->last_push_child + 1) % component->nchildren ;
	ws->last_push_child = i;
	return i;
}


/**
 * Return a worker from which a task can be stolen.
 * This is a phony function used to call the right
 * function depending on the value of USE_OVERLOAD.
 */
static inline struct starpu_task * steal_task(struct starpu_sched_component * component, int workerid)
{
	return steal_task_round_robin(component, workerid);
}

/**
 * Return a worker from which a task can be stolen.
 * This is a phony function used to call the right
 * function depending on the value of USE_OVERLOAD.
 */
static inline unsigned select_worker(struct starpu_sched_component * component)
{
	return select_worker_round_robin(component);
}


static int is_worker_of_component(struct starpu_sched_component * component, int workerid)
{
	return starpu_bitmap_get(component->workers, workerid);
}



static struct starpu_task * pull_task(struct starpu_sched_component * component, struct starpu_sched_component * to STARPU_ATTRIBUTE_UNUSED)
{
	unsigned workerid = starpu_worker_get_id_check();
	unsigned i;
	for(i = 0; i < component->nchildren; i++)
	{
		if(is_worker_of_component(component->children[i], workerid))
			break;
	}
	STARPU_ASSERT(i < component->nchildren);
	struct _starpu_work_stealing_data * wsd = component->data;
	const double now = starpu_timing_now();
	STARPU_COMPONENT_MUTEX_LOCK(wsd->mutexes[i]);
	struct starpu_task * task = _starpu_prio_deque_pop_task(wsd->fifos[i]);
	if(task)
	{
		if(!isnan(task->predicted))
		{
			wsd->fifos[i]->exp_len -= task->predicted;
			wsd->fifos[i]->exp_start = now + task->predicted;
		}
	}
	else
		wsd->fifos[i]->exp_len = 0.0;

	STARPU_COMPONENT_MUTEX_UNLOCK(wsd->mutexes[i]);
	if(task)
	{
		return task;
	}

	task  = steal_task(component, workerid);
	if(task)
	{
		STARPU_COMPONENT_MUTEX_LOCK(wsd->mutexes[i]);
		wsd->fifos[i]->nprocessed++;
		STARPU_COMPONENT_MUTEX_UNLOCK(wsd->mutexes[i]);

		return task;
	}
	for(i=0; i < component->nparents; i++)
	{
		if(component->parents[i] == NULL)
			continue;
		else
		{
			task = starpu_sched_component_pull_task(component->parents[i],component);
			if(task)
				break;
		}
	}
	if(task)
		return task;
	else
		return NULL;
}

double _ws_estimated_end(struct starpu_sched_component * component)
{
	STARPU_ASSERT(starpu_sched_component_is_work_stealing(component));
	struct _starpu_work_stealing_data * wsd = component->data;
	double sum_len = 0.0;
	double sum_start = 0.0;
	unsigned i;
	const double now = starpu_timing_now();
	for(i = 0; i < component->nchildren; i++)
	{
		STARPU_COMPONENT_MUTEX_LOCK(wsd->mutexes[i]);
		sum_len += wsd->fifos[i]->exp_len;
		wsd->fifos[i]->exp_start = STARPU_MAX(now, wsd->fifos[i]->exp_start);
		sum_start += wsd->fifos[i]->exp_start;
		STARPU_COMPONENT_MUTEX_UNLOCK(wsd->mutexes[i]);

	}
	int nb_workers = starpu_bitmap_cardinal(component->workers_in_ctx);

	return (sum_start + sum_len) / nb_workers;
}

double _ws_estimated_load(struct starpu_sched_component * component)
{
	STARPU_ASSERT(starpu_sched_component_is_work_stealing(component));
	struct _starpu_work_stealing_data * wsd = component->data;
	int ntasks = 0;
	unsigned i;
	for(i = 0; i < component->nchildren; i++)
	{
		STARPU_COMPONENT_MUTEX_LOCK(wsd->mutexes[i]);
		ntasks += wsd->fifos[i]->ntasks;
		STARPU_COMPONENT_MUTEX_UNLOCK(wsd->mutexes[i]);
	}
	double speedup = 0.0;
	int workerid;
	for(workerid = starpu_bitmap_first(component->workers_in_ctx);
	    -1 != workerid;
	    workerid = starpu_bitmap_next(component->workers_in_ctx, workerid))
	{
		speedup += starpu_worker_get_relative_speedup(starpu_worker_get_perf_archtype(workerid, component->tree->sched_ctx_id));
	}

	return ntasks / speedup;
}

static int push_task(struct starpu_sched_component * component, struct starpu_task * task)
{
	struct _starpu_work_stealing_data * wsd = component->data;
	int ret;
	unsigned i = wsd->last_push_child;
	i = (i+1)%component->nchildren;
	STARPU_COMPONENT_MUTEX_LOCK(wsd->mutexes[i]);
	starpu_sched_task_break(task);
	ret = _starpu_prio_deque_push_front_task(wsd->fifos[i], task);
	STARPU_COMPONENT_MUTEX_UNLOCK(wsd->mutexes[i]);

	wsd->last_push_child = i;
	component->can_pull(component);
	return ret;
}


//this function is special, when a worker call it, we want to push the task in his fifo
int starpu_sched_tree_work_stealing_push_task(struct starpu_task *task)
{
	int workerid = starpu_worker_get_id();
	if(workerid == -1)
		return starpu_sched_tree_push_task(task);

	unsigned sched_ctx_id = task->sched_ctx;
	struct starpu_sched_component * component =starpu_sched_component_worker_get(sched_ctx_id, workerid);
	while(sched_ctx_id < component->nparents && component->parents[sched_ctx_id] != NULL)
	{
		component = component->parents[sched_ctx_id];
		if(starpu_sched_component_is_work_stealing(component))
		{
			if(!starpu_sched_component_can_execute_task(component, task))
				return starpu_sched_tree_push_task(task);

			unsigned i;
			for(i = 0; i < component->nchildren; i++)
				if(is_worker_of_component(component->children[i], workerid))
					break;
			STARPU_ASSERT(i < component->nchildren);

			struct _starpu_work_stealing_data * wsd = component->data;
			STARPU_COMPONENT_MUTEX_LOCK(wsd->mutexes[i]);
			int ret = _starpu_prio_deque_push_front_task(wsd->fifos[i] , task);
			if(ret == 0 && !isnan(task->predicted))
				wsd->fifos[i]->exp_len += task->predicted;
			STARPU_COMPONENT_MUTEX_UNLOCK(wsd->mutexes[i]);

			component->can_pull(component);
			return ret;
		}
	}

	return starpu_sched_tree_push_task(task);
}


void _ws_add_child(struct starpu_sched_component * component, struct starpu_sched_component * child)
{
	struct _starpu_work_stealing_data * wsd = component->data;
	starpu_sched_component_add_child(component, child);
	if(wsd->size < component->nchildren)
	{
		STARPU_ASSERT(wsd->size == component->nchildren - 1);
		_STARPU_REALLOC(wsd->fifos, component->nchildren * sizeof(*wsd->fifos));
		_STARPU_REALLOC(wsd->mutexes, component->nchildren * sizeof(*wsd->mutexes));
		wsd->size = component->nchildren;
	}

	struct _starpu_prio_deque *fifo;
	_STARPU_MALLOC(fifo, sizeof(*fifo));
	_starpu_prio_deque_init(fifo);
	wsd->fifos[component->nchildren - 1] = fifo;

	starpu_pthread_mutex_t *mutex;
	_STARPU_MALLOC(mutex, sizeof(*mutex));
	STARPU_PTHREAD_MUTEX_INIT(mutex,NULL);
	wsd->mutexes[component->nchildren - 1] = mutex;
}

void _ws_remove_child(struct starpu_sched_component * component, struct starpu_sched_component * child)
{
	struct _starpu_work_stealing_data * wsd = component->data;

	STARPU_PTHREAD_MUTEX_DESTROY(wsd->mutexes[component->nchildren - 1]);
	free(wsd->mutexes[component->nchildren - 1]);

	unsigned i_component;
	for(i_component = 0; i_component < component->nchildren; i_component++)
	{
		if(component->children[i_component] == child)
			break;
	}
	STARPU_ASSERT(i_component != component->nchildren);
	struct _starpu_prio_deque * tmp_fifo = wsd->fifos[i_component];
	wsd->fifos[i_component] = wsd->fifos[component->nchildren - 1];


	component->children[i_component] = component->children[component->nchildren - 1];
	component->nchildren--;
	struct starpu_task * task;
	while ((task = _starpu_prio_deque_pop_task(tmp_fifo)))
	{
		starpu_sched_component_push_task(NULL, component, task);
	}
	_starpu_prio_deque_destroy(tmp_fifo);
	free(tmp_fifo);
}

void _work_stealing_component_deinit_data(struct starpu_sched_component * component)
{
	struct _starpu_work_stealing_data * wsd = component->data;
	free(wsd->fifos);
	free(wsd->mutexes);
	free(wsd);
}

int starpu_sched_component_is_work_stealing(struct starpu_sched_component * component)
{
	return component->push_task == push_task;
}

struct starpu_sched_component * starpu_sched_component_work_stealing_create(struct starpu_sched_tree *tree, void *arg)
{
	(void)arg;
	struct starpu_sched_component *component = starpu_sched_component_create(tree, "work_stealing");
	struct _starpu_work_stealing_data *wsd;
	_STARPU_CALLOC(wsd, 1, sizeof(*wsd));
	component->pull_task = pull_task;
	component->push_task = push_task;
	component->add_child = _ws_add_child;
	component->remove_child = _ws_remove_child;
	component->estimated_end = _ws_estimated_end;
	component->estimated_load = _ws_estimated_load;
	component->deinit_data = _work_stealing_component_deinit_data;
	component->data = wsd;
	return  component;
}
