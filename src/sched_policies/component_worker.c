/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2011       Télécom-SudParis
 * Copyright (C) 2013       Simon Archipoff
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

#include <starpu_sched_component.h>
#include <sched_policies/sched_component.h>
#include <core/workers.h>

#include <float.h>

/* data structure for worker's queue look like this :
 * W = worker
 * T = simple task
 * P = parallel task
 *
 *
 *         P--P  T
 *         |  | \|
 *   P--P  T  T  P  T
 *   |  |  |  |  |  |
 *   T  T  P--P--P  T
 *   |  |  |  |  |  |
 *   W  W  W  W  W  W
 *
 *
 *
 * its possible that a _starpu_task_grid wont have task, because it have been
 * poped by a worker.
 *
 * N = no task
 *
 *   T  T  T
 *   |  |  |
 *   P--N--N
 *   |  |  |
 *   W  W  W
 *
 *
 * this API is a little asymmetric : struct _starpu_task_grid are allocated by the caller and freed by the data structure
 *
 */



/******************************************************************************
 *					  Worker Components' Data Structures					  *
 *****************************************************************************/



struct _starpu_task_grid
{
	/* this member may be NULL if a worker have poped it but its a
	 * parallel task and we dont want mad pointers
	 */
	struct starpu_task * task;

	struct _starpu_task_grid *up, *down, *left, *right;

	/* this is used to count the number of task to be poped by a worker
	 * the leftist _starpu_task_grid maintain the ntasks counter (ie .left == NULL),
	 * all the others use the pntasks that point to it
	 *
	 * when the counter reach 0, all the left and right member are set to NULL,
	 * that mean that we will free that components.
	 */
	union
	{
		int ntasks;
		int * pntasks;
	};
};


/* list->exp_start, list->exp_len, list-exp_end and list->ntasks
 * are updated by starpu_sched_component_worker_push_task(component, task) and pre_exec_hook
 */
struct _starpu_worker_task_list
{
	double exp_start, exp_len, exp_end, pipeline_len;
	struct _starpu_task_grid *first, *last;
	unsigned ntasks;
	starpu_pthread_mutex_t mutex;
};

/* This is called when a transfer request is actually pushed to the worker */
static void _starpu_worker_task_list_transfer_started(struct _starpu_worker_task_list *l, struct starpu_task *task)
{
	double transfer_model = task->predicted_transfer;
	if (isnan(transfer_model))
		return;

	/* We now start the transfer, move it from predicted to pipelined */
	l->exp_len -= transfer_model;
	l->pipeline_len += transfer_model;
	l->exp_start = starpu_timing_now() + l->pipeline_len;
	l->exp_end = l->exp_start + l->exp_len;
}

#ifdef STARPU_DEVEL
#warning FIXME: merge with deque_modeling_policy_data_aware
#endif
/* This is called when a task is actually pushed to the worker (i.e. the transfer finished */
static void _starpu_worker_task_list_started(struct _starpu_worker_task_list *l, struct starpu_task *task)
{
	double model = task->predicted;
	double transfer_model = task->predicted_transfer;
	if(!isnan(transfer_model))
		/* The transfer is over, remove it from pipelined */
		l->pipeline_len -= transfer_model;

	if(!isnan(model))
	{
		/* We now start the computation, move it from predicted to pipelined */
		l->exp_len -= model;
		l->pipeline_len += model;
		l->exp_start = starpu_timing_now() + l->pipeline_len;
                l->exp_end= l->exp_start + l->exp_len;
	}
}

/* This is called when a task is actually finished */
static void _starpu_worker_task_list_finished(struct _starpu_worker_task_list *l, struct starpu_task *task)
{
	if(!isnan(task->predicted))
		/* The execution is over, remove it from pipelined */
		l->pipeline_len -= task->predicted;
	l->exp_start = STARPU_MAX(starpu_timing_now() + l->pipeline_len, l->exp_start);
	l->exp_end = l->exp_start + l->exp_len;
}


struct _starpu_worker_component_data
{
	union
	{
		struct _starpu_worker * worker;
		struct
		{
			unsigned worker_size;
			unsigned workerids[STARPU_NMAXWORKERS];
		} parallel_worker;
	};
	struct _starpu_worker_task_list * list;
};

/* this array store worker components */
static struct starpu_sched_component * _worker_components[STARPU_NMAX_SCHED_CTXS][STARPU_NMAXWORKERS];


/******************************************************************************
 *				Worker Components' Task List and Grid Functions				  *
 *****************************************************************************/



static struct _starpu_worker_task_list * _starpu_worker_task_list_create(void)
{
	struct _starpu_worker_task_list *l;
	_STARPU_MALLOC(l, sizeof(*l));
	memset(l, 0, sizeof(*l));
	l->exp_len = l->pipeline_len = 0.0;
	l->exp_start = l->exp_end = starpu_timing_now();
	/* These are only for statistics */
	STARPU_HG_DISABLE_CHECKING(l->exp_end);
	STARPU_HG_DISABLE_CHECKING(l->exp_start);
	STARPU_PTHREAD_MUTEX_INIT(&l->mutex,NULL);
	return l;
}

static struct _starpu_task_grid * _starpu_task_grid_create(void)
{
	struct _starpu_task_grid *t;
	_STARPU_MALLOC(t, sizeof(*t));
	memset(t, 0, sizeof(*t));
	return t;
}

static struct _starpu_worker_task_list * _worker_get_list(unsigned sched_ctx_id)
{
	unsigned workerid = starpu_worker_get_id_check();
	STARPU_ASSERT(workerid < starpu_worker_get_count());
	struct _starpu_worker_component_data * d = starpu_sched_component_worker_get(sched_ctx_id, workerid)->data;
	return d->list;
}

static void _starpu_task_grid_destroy(struct _starpu_task_grid * t)
{
	free(t);
}

static void _starpu_worker_task_list_destroy(struct _starpu_worker_task_list * l)
{
	if(l)
	{
		/* There can be empty task grids, when we picked the last task after the front task grid */
		struct _starpu_task_grid *t = l->first, *nextt;

		while(t)
		{
			STARPU_ASSERT(!t->task);
			nextt = t->up;
			_starpu_task_grid_destroy(t);
			t = nextt;
		}
		STARPU_PTHREAD_MUTEX_DESTROY(&l->mutex);
		free(l);
	}
}

static inline void _starpu_worker_task_list_add(struct _starpu_worker_task_list * l, struct starpu_task *task)
{
	double predicted = task->predicted;
	double predicted_transfer = task->predicted_transfer;
	double end = l->exp_end;
	const double now = starpu_timing_now();

	/* Sometimes workers didn't take the tasks as early as we expected */
	l->exp_start = STARPU_MAX(l->exp_start, now);

	if (now + predicted_transfer < end)
	{
		/* We may hope that the transfer will be finished by
		 * the start of the task. */
		predicted_transfer = 0.0;
	}
	else
	{
		/* The transfer will not be finished by then, take the
		 * remainder into account */
		predicted_transfer = (now + predicted_transfer) - end;
	}

	if(!isnan(predicted_transfer))
		l->exp_len += predicted_transfer;

	if(!isnan(predicted))
		l->exp_len += predicted;

	l->exp_end = l->exp_start + l->exp_len;

	task->predicted = predicted;
	task->predicted_transfer = predicted_transfer;
}

static inline void _starpu_worker_task_list_push(struct _starpu_worker_task_list * l, struct _starpu_task_grid * t)
{
/* the task, ntasks, pntasks, left and right members of t are set by the caller */
	STARPU_ASSERT(t->task);
	if(l->first == NULL)
		l->first = l->last = t;
	t->down = l->last;
	l->last->up = t;
	t->up = NULL;
	l->last = t;
	l->ntasks++;

	_starpu_worker_task_list_add(l, t->task);
}

/* recursively set left and right pointers to NULL */
static inline void _starpu_task_grid_unset_left_right_member(struct _starpu_task_grid * t)
{
	STARPU_ASSERT(t->task == NULL);
	struct _starpu_task_grid * t_left = t->left;
	struct _starpu_task_grid * t_right = t->right;
	t->left = t->right = NULL;
	while(t_left)
	{
		STARPU_ASSERT(t_left->task == NULL);
		t = t_left;
		t_left = t_left->left;
		t->left = NULL;
		t->right = NULL;
	}
	while(t_right)
	{
		STARPU_ASSERT(t_right->task == NULL);
		t = t_right;
		t_right = t_right->right;
		t->left = NULL;
		t->right = NULL;
	}
}

static inline struct starpu_task * _starpu_worker_task_list_pop(struct _starpu_worker_task_list * l)
{
 	if(!l->first)
	{
		l->exp_len = l->pipeline_len = 0.0;
		l->exp_start = l->exp_end = starpu_timing_now();
		return NULL;
	}
	struct _starpu_task_grid * t = l->first;

	/* if there is no task there is no tasks linked to this, then we can free it */
	if(t->task == NULL && t->right == NULL && t->left == NULL)
	{
		l->first = t->up;
		if(l->first)
			l->first->down = NULL;
		if(l->last == t)
			l->last = NULL;
		_starpu_task_grid_destroy(t);
		return _starpu_worker_task_list_pop(l);
	}

	while(t)
	{
		if(t->task)
		{
			struct starpu_task * task = t->task;
			t->task = NULL;
			/* the leftist thing hold the number of tasks, other have a pointer to it */
			int * p = t->left ? t->pntasks : &t->ntasks;

			/* the worker who pop the last task allow the rope to be freed */
			if(STARPU_ATOMIC_ADD(p, -1) == 0)
				_starpu_task_grid_unset_left_right_member(t);

			l->ntasks--;

			return task;
		}
		t = t->up;
	}

	return NULL;
}



/******************************************************************************
 *			Worker Components' Public Helper Functions (Part 1)		     	  *
 *****************************************************************************/



struct _starpu_worker * _starpu_sched_component_worker_get_worker(struct starpu_sched_component * worker_component)
{
	STARPU_ASSERT(starpu_sched_component_is_simple_worker(worker_component));
	struct _starpu_worker_component_data * data = worker_component->data;
	return data->worker;
}

/******************************************************************************
 *				Worker Components' Private Helper Functions			      	  *
 *****************************************************************************/



#ifndef STARPU_NO_ASSERT
static int _worker_consistant(struct starpu_sched_component * component)
{
	int is_a_worker = 0;
	int i;
	for(i = 0; i<STARPU_NMAXWORKERS; i++)
		if(_worker_components[component->tree->sched_ctx_id][i] == component)
			is_a_worker = 1;
	if(!is_a_worker)
		return 0;
	struct _starpu_worker_component_data * data = component->data;
	if(data->worker)
	{
		int id = data->worker->workerid;
		return  (_worker_components[component->tree->sched_ctx_id][id] == component)
			&&  component->nchildren == 0;
	}
	return 1;
}
#endif



/******************************************************************************
 *				Simple Worker Components' Interface Functions			      *
 *****************************************************************************/



static int simple_worker_can_pull(struct starpu_sched_component * worker_component)
{
	struct _starpu_worker * worker = _starpu_sched_component_worker_get_worker(worker_component);
	int workerid = worker->workerid;
	return starpu_wake_worker_relax_light(workerid);
}

static int simple_worker_push_task(struct starpu_sched_component * component, struct starpu_task *task)
{
	STARPU_ASSERT(starpu_sched_component_is_worker(component));
	/*this function take the worker's mutex */
	struct _starpu_worker_component_data * data = component->data;
	struct _starpu_task_grid * t = _starpu_task_grid_create();
	t->task = task;
	t->ntasks = 1;

	task->workerid = starpu_bitmap_first(component->workers);
#if 1 /* dead lock problem? */
	if (starpu_get_prefetch_flag() && !task->prefetched)
		starpu_prefetch_task_input_for(task, task->workerid);
#endif
	struct _starpu_worker_task_list * list = data->list;
	STARPU_COMPONENT_MUTEX_LOCK(&list->mutex);
	_starpu_worker_task_list_push(list, t);
	STARPU_COMPONENT_MUTEX_UNLOCK(&list->mutex);
	simple_worker_can_pull(component);
	return 0;
}

static struct starpu_task * simple_worker_pull_task(struct starpu_sched_component *component, struct starpu_sched_component * to)
{
	unsigned workerid = starpu_worker_get_id_check();
	struct _starpu_worker *worker = _starpu_get_worker_struct(workerid);
	struct _starpu_worker_component_data * data = component->data;
	struct _starpu_worker_task_list * list = data->list;
	struct starpu_task * task;
	unsigned i;
	int n_tries = 0;
	do
	{
		const double now = starpu_timing_now();
		/* do not reset state_keep_awake here has it may hide tasks in worker->local_tasks */
		n_tries++;
		STARPU_COMPONENT_MUTEX_LOCK(&list->mutex);
		/* Take the opportunity to update start time */
		data->list->exp_start = STARPU_MAX(now, data->list->exp_start);
		data->list->exp_end = data->list->exp_start + data->list->exp_len;
		task =  _starpu_worker_task_list_pop(list);
		if(task)
		{
			_starpu_worker_task_list_transfer_started(list, task);
			STARPU_COMPONENT_MUTEX_UNLOCK(&list->mutex);
			starpu_push_task_end(task);
			goto ret;
		}
		STARPU_COMPONENT_MUTEX_UNLOCK(&list->mutex);
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
	}
	while((!task) && worker->state_keep_awake && n_tries < 2);
	if(!task)
		goto ret;
	if(task->cl->type == STARPU_SPMD)
	{
		if(!starpu_worker_is_combined_worker(workerid))
		{
			STARPU_COMPONENT_MUTEX_LOCK(&list->mutex);
			_starpu_worker_task_list_add(list, task);
			_starpu_worker_task_list_transfer_started(list, task);
			STARPU_COMPONENT_MUTEX_UNLOCK(&list->mutex);
			starpu_push_task_end(task);
			goto ret;
		}
		struct starpu_sched_component * combined_worker_component = starpu_sched_component_worker_get(component->tree->sched_ctx_id, workerid);
		starpu_sched_component_push_task(component, combined_worker_component, task);
		/* we have pushed a task in queue, so can make a recursive call */
		task = simple_worker_pull_task(component, to);
		goto ret;

	}
	if(task)
	{
		STARPU_COMPONENT_MUTEX_LOCK(&list->mutex);
		_starpu_worker_task_list_add(list, task);
		_starpu_worker_task_list_transfer_started(list, task);
		STARPU_COMPONENT_MUTEX_UNLOCK(&list->mutex);
		starpu_push_task_end(task);
	}
ret:
	return task;
}

static double simple_worker_estimated_end(struct starpu_sched_component * component)
{
	struct _starpu_worker_component_data * data = component->data;
	double now = starpu_timing_now();
	if (now + data->list->pipeline_len > data->list->exp_start )
	{
		data->list->exp_start = now + data->list->pipeline_len;
		data->list->exp_end = data->list->exp_start + data->list->exp_len;
	}
	return data->list->exp_end;
}

static double simple_worker_estimated_load(struct starpu_sched_component * component)
{
	struct _starpu_worker * worker = _starpu_sched_component_worker_get_worker(component);
	int nb_task = 0;
	STARPU_COMPONENT_MUTEX_LOCK(&worker->mutex);
	struct starpu_task_list list = worker->local_tasks;
	struct starpu_task * task;
	for(task = starpu_task_list_front(&list);
	    task != starpu_task_list_end(&list);
	    task = starpu_task_list_next(task))
		nb_task++;
	STARPU_COMPONENT_MUTEX_UNLOCK(&worker->mutex);
	struct _starpu_worker_component_data * d = component->data;
	struct _starpu_worker_task_list * l = d->list;
	int ntasks_in_fifo = l ? l->ntasks : 0;
	return (double) (nb_task + ntasks_in_fifo)
		/ starpu_worker_get_relative_speedup(
				starpu_worker_get_perf_archtype(starpu_bitmap_first(component->workers), component->tree->sched_ctx_id));
}

static void _worker_component_deinit_data(struct starpu_sched_component * component)
{
	struct _starpu_worker_component_data * d = component->data;
	_starpu_worker_task_list_destroy(d->list);
	int i, j;
	for(j = 0; j < STARPU_NMAX_SCHED_CTXS; j++)
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
		if(_worker_components[j][i] == component)
		{
			_worker_components[j][i] = NULL;
			break;
		}
	free(d);
}

static struct starpu_sched_component * starpu_sched_component_worker_create(struct starpu_sched_tree *tree, int workerid)
{
	STARPU_ASSERT(workerid >= 0 && workerid < (int) starpu_worker_get_count());

	if(_worker_components[tree->sched_ctx_id][workerid])
		return _worker_components[tree->sched_ctx_id][workerid];

	struct _starpu_worker * worker = _starpu_get_worker_struct(workerid);
	if(worker == NULL)
		return NULL;
	char name[32];
	snprintf(name, sizeof(name), "worker %d", workerid);
	struct starpu_sched_component * component = starpu_sched_component_create(tree, name);
	struct _starpu_worker_component_data *data;
	_STARPU_MALLOC(data, sizeof(*data));
	memset(data, 0, sizeof(*data));

	data->worker = worker;
	data->list = _starpu_worker_task_list_create();
	component->data = data;

	component->push_task = simple_worker_push_task;
	component->pull_task = simple_worker_pull_task;
	component->can_pull = simple_worker_can_pull;
	component->estimated_end = simple_worker_estimated_end;
	component->estimated_load = simple_worker_estimated_load;
	component->deinit_data = _worker_component_deinit_data;
	starpu_bitmap_set(component->workers, workerid);
	starpu_bitmap_or(component->workers_in_ctx, component->workers);
	_worker_components[tree->sched_ctx_id][workerid] = component;

	/*
#ifdef STARPU_HAVE_HWLOC
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	struct _starpu_machine_topology *topology = &config->topology;
	hwloc_obj_t obj = hwloc_get_obj_by_depth(topology->hwtopology, config->pu_depth, worker->bindid);
	STARPU_ASSERT(obj);
	component->obj = obj;
#endif
	*/

	return component;
}



/******************************************************************************
 *				Combined Worker Components' Interface Functions			      *
 *****************************************************************************/



static int combined_worker_can_pull(struct starpu_sched_component * component)
{
	(void) component;
	STARPU_ASSERT(starpu_sched_component_is_combined_worker(component));
	struct _starpu_worker_component_data * data = component->data;
	int workerid = starpu_worker_get_id();
	unsigned i;
	for(i = 0; i < data->parallel_worker.worker_size; i++)
	{
		int target = data->parallel_worker.workerids[i];
		if(target == workerid)
			continue;
		if (starpu_wake_worker_relax_light(target))
			return 1;
	}
	return 0;
}

static int combined_worker_push_task(struct starpu_sched_component * component, struct starpu_task *task)
{
	STARPU_ASSERT(starpu_sched_component_is_combined_worker(component));
	struct _starpu_worker_component_data * data = component->data;
	STARPU_ASSERT(data->parallel_worker.worker_size >= 1);
	struct _starpu_task_grid * task_alias[data->parallel_worker.worker_size];
	starpu_parallel_task_barrier_init(task, starpu_bitmap_first(component->workers));
	task_alias[0] = _starpu_task_grid_create();
	task_alias[0]->task = starpu_task_dup(task);
	task_alias[0]->task->workerid = data->parallel_worker.workerids[0];
	task_alias[0]->task->destroy = 1;
	task_alias[0]->left = NULL;
	task_alias[0]->ntasks = data->parallel_worker.worker_size;
	_STARPU_TRACE_JOB_PUSH(task_alias[0]->task, task_alias[0]->task->priority > 0);
	unsigned i;
	for(i = 1; i < data->parallel_worker.worker_size; i++)
	{
		task_alias[i] = _starpu_task_grid_create();
		task_alias[i]->task = starpu_task_dup(task);
		task_alias[i]->task->destroy = 1;
		task_alias[i]->task->workerid = data->parallel_worker.workerids[i];
		task_alias[i]->left = task_alias[i-1];
		task_alias[i - 1]->right = task_alias[i];
		task_alias[i]->pntasks = &(task_alias[0]->ntasks);
		_STARPU_TRACE_JOB_PUSH(task_alias[i]->task, task_alias[i]->task->priority > 0);
	}

	starpu_pthread_mutex_t * mutex_to_unlock = NULL;
	i = 0;
	do
	{
		struct starpu_sched_component * worker_component = starpu_sched_component_worker_get(component->tree->sched_ctx_id, data->parallel_worker.workerids[i]);
		struct _starpu_worker_component_data * worker_data = worker_component->data;
		struct _starpu_worker_task_list * list = worker_data->list;
		STARPU_COMPONENT_MUTEX_LOCK(&list->mutex);
		if(mutex_to_unlock)
			STARPU_COMPONENT_MUTEX_UNLOCK(mutex_to_unlock);
		mutex_to_unlock = &list->mutex;

		_starpu_worker_task_list_push(list, task_alias[i]);
		i++;
	}
	while(i < data->parallel_worker.worker_size);

	STARPU_COMPONENT_MUTEX_UNLOCK(mutex_to_unlock);

	int workerid = starpu_worker_get_id();
	if(-1 == workerid)
	{
		combined_worker_can_pull(component);
	}
	else
	{
		/* wake up all other workers of combined worker */
		for(i = 0; i < data->parallel_worker.worker_size; i++)
		{
			struct starpu_sched_component * worker_component = starpu_sched_component_worker_get(component->tree->sched_ctx_id, data->parallel_worker.workerids[i]);
			simple_worker_can_pull(worker_component);
		}

		combined_worker_can_pull(component);
	}

	return 0;
}

static struct starpu_task *combined_worker_pull_task(struct starpu_sched_component * from STARPU_ATTRIBUTE_UNUSED, struct starpu_sched_component * to STARPU_ATTRIBUTE_UNUSED)
{
	return NULL;
}

static double combined_worker_estimated_end(struct starpu_sched_component * component)
{
	STARPU_ASSERT(starpu_sched_component_is_combined_worker(component));
	struct _starpu_worker_component_data * d = component->data;
	double max = 0.0;
	unsigned i;
	for(i = 0; i < d->parallel_worker.worker_size; i++)
	{
		struct _starpu_worker_component_data * data;
		data = _worker_components[component->tree->sched_ctx_id][d->parallel_worker.workerids[i]]->data;
		double tmp = data->list->exp_end;
		max = tmp > max ? tmp : max;
	}
	return max;
}

static double combined_worker_estimated_load(struct starpu_sched_component * component)
{
	struct _starpu_worker_component_data * d = component->data;
	double load = 0;
	unsigned i;
	for(i = 0; i < d->parallel_worker.worker_size; i++)
	{
		struct starpu_sched_component * n = starpu_sched_component_worker_get(component->tree->sched_ctx_id, d->parallel_worker.workerids[i]);
		load += n->estimated_load(n);
	}
	return load;
}

struct starpu_sched_component *starpu_sched_component_parallel_worker_create(struct starpu_sched_tree *tree, unsigned nworkers, unsigned *workers)
{
	struct starpu_sched_component * component = starpu_sched_component_create(tree, "combined_worker");

	struct _starpu_worker_component_data *data;
	_STARPU_MALLOC(data, sizeof(*data));
	memset(data, 0, sizeof(*data));
	STARPU_ASSERT(nworkers <= STARPU_NMAXWORKERS);
	STARPU_ASSERT(nworkers <= starpu_worker_get_count());
	data->parallel_worker.worker_size = nworkers;
	memcpy(data->parallel_worker.workerids, workers, nworkers * sizeof(unsigned));

	component->data = data;
	component->push_task = combined_worker_push_task;
	component->pull_task = combined_worker_pull_task;
	component->estimated_end = combined_worker_estimated_end;
	component->estimated_load = combined_worker_estimated_load;
	component->can_pull = combined_worker_can_pull;
	component->deinit_data = _worker_component_deinit_data;
	
	unsigned i;
	for (i = 0; i < nworkers; i++)
		starpu_sched_component_connect(component, starpu_sched_component_worker_get(tree->sched_ctx_id, workers[i]));

	return component;
}

static struct starpu_sched_component  * starpu_sched_component_combined_worker_create(struct starpu_sched_tree *tree, int workerid)
{
	STARPU_ASSERT(workerid >= 0 && workerid <  STARPU_NMAXWORKERS);

	if(_worker_components[tree->sched_ctx_id][workerid])
		return _worker_components[tree->sched_ctx_id][workerid];

	struct _starpu_combined_worker * combined_worker = _starpu_get_combined_worker_struct(workerid);
	if(combined_worker == NULL)
		return NULL;

	struct starpu_sched_component *component = starpu_sched_component_parallel_worker_create(tree, combined_worker->worker_size, (unsigned *) combined_worker->combined_workerid);

	starpu_bitmap_set(component->workers, workerid);
	starpu_bitmap_or(component->workers_in_ctx, component->workers);

	_worker_components[tree->sched_ctx_id][workerid] = component;

	/*
#ifdef STARPU_HAVE_HWLOC
	struct _starpu_worker_component_data * data = component->data;
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	struct _starpu_machine_topology *topology = &config->topology;
	struct _starpu_worker *worker = _starpu_get_worker_struct(data->parallel_worker.workerids[0]);
	hwloc_obj_t obj = hwloc_get_obj_by_depth(topology->hwtopology, config->pu_depth, worker->bindid);
	STARPU_ASSERT(obj);
	component->obj = obj;
#endif
	*/
	return component;
}


/******************************************************************************
 *			Worker Components' Public Helper Functions (Part 2)			      *
 *****************************************************************************/



void _starpu_sched_component_lock_all_workers(void)
{
	unsigned i;
	for(i = 0; i < starpu_worker_get_count(); i++)
		starpu_worker_lock(i);
}
void _starpu_sched_component_unlock_all_workers(void)
{
	unsigned i;
	for(i = 0; i < starpu_worker_get_count(); i++)
		starpu_worker_unlock(i);
}

void _starpu_sched_component_workers_destroy(void)
{
	int i, j;
	for(j = 0; j < STARPU_NMAX_SCHED_CTXS; j++)
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
		if (_worker_components[j][i])
			starpu_sched_component_destroy(_worker_components[j][i]);
}

int starpu_sched_component_worker_get_workerid(struct starpu_sched_component * worker_component)
{
#ifndef STARPU_NO_ASSERT
	STARPU_ASSERT(_worker_consistant(worker_component));
#endif
	STARPU_ASSERT(1 == starpu_bitmap_cardinal(worker_component->workers));
	return starpu_bitmap_first(worker_component->workers);
}

void starpu_sched_component_worker_pre_exec_hook(struct starpu_task * task, unsigned sched_ctx_id STARPU_ATTRIBUTE_UNUSED)
{
	struct _starpu_worker_task_list * list = _worker_get_list(sched_ctx_id);
	const double now = starpu_timing_now();
	STARPU_COMPONENT_MUTEX_LOCK(&list->mutex);
	_starpu_worker_task_list_started(list, task);
	/* Take the opportunity to update start time */
	list->exp_start = STARPU_MAX(now + list->pipeline_len, list->exp_start);
	STARPU_COMPONENT_MUTEX_UNLOCK(&list->mutex);
}

void starpu_sched_component_worker_post_exec_hook(struct starpu_task * task, unsigned sched_ctx_id STARPU_ATTRIBUTE_UNUSED)
{
	if(task->execute_on_a_specific_worker)
		return;
	struct _starpu_worker_task_list * list = _worker_get_list(sched_ctx_id);
	STARPU_COMPONENT_MUTEX_LOCK(&list->mutex);
	_starpu_worker_task_list_finished(list, task);
	STARPU_COMPONENT_MUTEX_UNLOCK(&list->mutex);
}

int starpu_sched_component_is_simple_worker(struct starpu_sched_component * component)
{
	return component->push_task == simple_worker_push_task;
}
int starpu_sched_component_is_combined_worker(struct starpu_sched_component * component)
{
	return component->push_task == combined_worker_push_task;
}

int starpu_sched_component_is_worker(struct starpu_sched_component * component)
{
	return starpu_sched_component_is_simple_worker(component)
		|| starpu_sched_component_is_combined_worker(component);
}

/* As Worker Components' creating functions are protected, this function allows
 * the user to get a Worker Component from a worker id */
struct starpu_sched_component * starpu_sched_component_worker_get(unsigned sched_ctx, int workerid)
{
	STARPU_ASSERT(workerid >= 0 && workerid < STARPU_NMAXWORKERS);
	/* we may need to take a mutex here */
	if (!_worker_components[sched_ctx][workerid])
		return starpu_sched_component_worker_new(sched_ctx, workerid);
	return _worker_components[sched_ctx][workerid];
}

struct starpu_sched_component * starpu_sched_component_worker_new(unsigned sched_ctx, int workerid)
{
	STARPU_ASSERT(workerid >= 0 && workerid < STARPU_NMAXWORKERS);
	/* we may need to take a mutex here */
	if (_worker_components[sched_ctx][workerid])
		return _worker_components[sched_ctx][workerid];
	struct starpu_sched_component * component;
	if(workerid < (int) starpu_worker_get_count())
		component = starpu_sched_component_worker_create(starpu_sched_tree_get(sched_ctx), workerid);
	else
		component = starpu_sched_component_combined_worker_create(starpu_sched_tree_get(sched_ctx), workerid);
	_worker_components[sched_ctx][workerid] = component;
	return component;
}




