/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <core/jobs.h>
#include <core/workers.h>
#include <starpu_sched_component.h>
#include <starpu_thread_util.h>

#include <float.h>

#include "sched_component.h"



/******************************************************************************
 *				Generic Scheduling Components' helper functions        		  *
 ******************************************************************************/



/* this function find the best implementation or an implementation that need to be calibrated for a worker available
 * and set prediction in *length. nan if a implementation need to be calibrated, 0.0 if no perf model are available
 * return false if no worker on the component can execute that task
 */
int starpu_sched_component_execute_preds(struct starpu_sched_component * component, struct starpu_task * task, double * length)
{
	STARPU_ASSERT(component && task);
	int can_execute = 0;
	starpu_task_bundle_t bundle = task->bundle;
	double len = DBL_MAX;


	int workerid;
	for(workerid = starpu_bitmap_first(component->workers_in_ctx);
	    workerid != -1;
	    workerid = starpu_bitmap_next(component->workers_in_ctx, workerid))
	{
		struct starpu_perfmodel_arch* archtype = starpu_worker_get_perf_archtype(workerid, component->tree->sched_ctx_id);
		int nimpl;
		for(nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		{
			if(starpu_worker_can_execute_task(workerid,task,nimpl)
			   || starpu_combined_worker_can_execute_task(workerid, task, nimpl))
			{
				double d;
				can_execute = 1;
				if(bundle)
					d = starpu_task_bundle_expected_length(bundle, archtype, nimpl);
				else
					d = starpu_task_expected_length(task, archtype, nimpl);
				if(isnan(d))
				{
					*length = d;
					return can_execute;
				}
				if(_STARPU_IS_ZERO(d))
				{
					continue;
				}
				STARPU_ASSERT_MSG(d >= 0, "workerid=%d, nimpl=%d, bundle=%p, d=%lf\n", workerid, nimpl, bundle, d);
				if(d < len)
				{
					len = d;
				}
			}
		}
		if(STARPU_SCHED_COMPONENT_IS_HOMOGENEOUS(component))
			break;
	}

	if(len == DBL_MAX) /* we dont have perf model */
		len = 0.0;
	if(length)
		*length = len;
	return can_execute;
}

/* very similar function that dont compute prediction */
int starpu_sched_component_can_execute_task(struct starpu_sched_component * component, struct starpu_task * task)
{
	STARPU_ASSERT(task);
	STARPU_ASSERT(component);
	unsigned nimpl;
	int worker;
	for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		for(worker = starpu_bitmap_first(component->workers_in_ctx);
		    -1 != worker;
		    worker = starpu_bitmap_next(component->workers_in_ctx, worker))
			if (starpu_worker_can_execute_task(worker, task, nimpl)
			     || starpu_combined_worker_can_execute_task(worker, task, nimpl))
			    return 1;
	return 0;
}

/* compute the average of transfer length for tasks on all workers
 * maybe this should be optimised if all workers are under the same numa component
 */
double starpu_sched_component_transfer_length(struct starpu_sched_component * component, struct starpu_task * task)
{
	STARPU_ASSERT(component && task);
	int nworkers = starpu_bitmap_cardinal(component->workers_in_ctx);
	double sum = 0.0;
	int worker;
	if(STARPU_SCHED_COMPONENT_IS_SINGLE_MEMORY_NODE(component))
	{
		unsigned memory_node  = starpu_worker_get_memory_node(starpu_bitmap_first(component->workers_in_ctx));
		if(task->bundle)
			return starpu_task_bundle_expected_data_transfer_time(task->bundle,memory_node);
		else
			return starpu_task_expected_data_transfer_time(memory_node, task);
	}

	for(worker = starpu_bitmap_first(component->workers_in_ctx);
	    worker != -1;
	    worker = starpu_bitmap_next(component->workers_in_ctx, worker))
	{
		unsigned memory_node  = starpu_worker_get_memory_node(worker);
		if(task->bundle)
		{
			sum += starpu_task_bundle_expected_data_transfer_time(task->bundle,memory_node);
		}
		else
		{
			sum += starpu_task_expected_data_transfer_time(memory_node, task);
			/* sum += starpu_task_expected_conversion_time(task, starpu_worker_get_perf_archtype(worker, component->tree->sched_ctx_id), impl ?)
			 * I dont know what to do as we dont know what implementation would be used here...
			 */
		}
	}
	return sum / nworkers;
}

/* This function can be called by components when they think that a prefetching request can be submitted.
 * For example, it is currently used by the MCT component to begin the prefetching on accelerators
 * on which it pushed tasks as soon as possible.
 */
void starpu_sched_component_prefetch_on_node(struct starpu_sched_component * component, struct starpu_task * task)
{
	if (starpu_get_prefetch_flag() && (!task->prefetched)
		&& (component->properties & STARPU_SCHED_COMPONENT_SINGLE_MEMORY_NODE))
	{
		int worker = starpu_bitmap_first(component->workers_in_ctx);
		unsigned memory_node = starpu_worker_get_memory_node(worker);
		starpu_prefetch_task_input_on_node(task, memory_node);
		task->prefetched = 1;
	}
}

/* remove all child
 * for all child of component, if child->parents[x] == component, set child->parents[x] to null
 * call component->deinit_data
 */
void starpu_sched_component_destroy(struct starpu_sched_component *component)
{
	STARPU_ASSERT(component);
	unsigned i,j;
	for(i = 0; i < component->nchildren; i++)
	{
		struct starpu_sched_component * child = component->children[i];
		for(j = 0; j < child->nparents; j++)
			if(child->parents[j] == component)
				child->remove_parent(child,component);

	}
	while(component->nchildren != 0)
		component->remove_child(component, component->children[0]);
	for(i = 0; i < component->nparents; i++)
	{
		struct starpu_sched_component * parent = component->parents[i];
		for(j = 0; j < parent->nchildren; j++)
			if(parent->children[j] == component)
				parent->remove_child(parent,component);

	}
	while(component->nparents != 0)
		component->remove_parent(component, component->parents[0]);
	component->deinit_data(component);
	free(component->children);
	free(component->parents);
	free(component->name);
	starpu_bitmap_destroy(component->workers);
	starpu_bitmap_destroy(component->workers_in_ctx);
	free(component);
}

void starpu_sched_component_destroy_rec(struct starpu_sched_component * component)
{
	if(component == NULL)
		return;

	unsigned i = 0;
	while(i < component->nchildren)
	{
		if (starpu_sched_component_is_worker(component->children[i]))
			i++;
		else
			starpu_sched_component_destroy_rec(component->children[i]);
	}

	if (!starpu_sched_component_is_worker(component))
		starpu_sched_component_destroy(component);
}

void set_properties(struct starpu_sched_component * component)
{
	STARPU_ASSERT(component);
	component->properties = 0;

	int worker = starpu_bitmap_first(component->workers_in_ctx);
	if (worker == -1)
		return;
	if (starpu_worker_is_combined_worker(worker))
		return;
#ifdef STARPU_DEVEL
#warning FIXME: Not all CUDA devices have the same speed
#endif
	uint32_t first_worker = _starpu_get_worker_struct(worker)->worker_mask;
	unsigned first_memory_node = _starpu_get_worker_struct(worker)->memory_node;
	int is_homogeneous = 1;
	int is_all_same_component = 1;
	for(;
	    worker != -1;
	    worker = starpu_bitmap_next(component->workers_in_ctx, worker))
	{
		if(starpu_worker_is_combined_worker(worker))
			continue;
		if(first_worker != _starpu_get_worker_struct(worker)->worker_mask)
			is_homogeneous = 0;
		if(first_memory_node != _starpu_get_worker_struct(worker)->memory_node)
			is_all_same_component = 0;
	}


	if(is_homogeneous)
		component->properties |= STARPU_SCHED_COMPONENT_HOMOGENEOUS;
	if(is_all_same_component)
		component->properties |= STARPU_SCHED_COMPONENT_SINGLE_MEMORY_NODE;
}


/* recursively set the component->workers member of component's subtree
 */
void _starpu_sched_component_update_workers(struct starpu_sched_component * component)
{
	STARPU_ASSERT(component);
	if(starpu_sched_component_is_worker(component))
		return;
	starpu_bitmap_unset_all(component->workers);
	unsigned i;
	for(i = 0; i < component->nchildren; i++)
	{
		_starpu_sched_component_update_workers(component->children[i]);
		starpu_bitmap_or(component->workers, component->children[i]->workers);
	}
	component->notify_change_workers(component);
}

/* recursively set the component->workers_in_ctx in component's subtree
 */
void _starpu_sched_component_update_workers_in_ctx(struct starpu_sched_component * component, unsigned sched_ctx_id)
{
	STARPU_ASSERT(component);
	/* worker components are shared among sched_ctxs, thus we do not apply the sched_ctx worker mask to them.
	 * per-ctx filtering is performed higher in the tree */
	if(starpu_sched_component_is_worker(component))
		return;
	struct starpu_bitmap * workers_in_ctx = _starpu_get_worker_mask(sched_ctx_id);
	starpu_bitmap_unset_and(component->workers_in_ctx,component->workers, workers_in_ctx);
	unsigned i,j;
	for(i = starpu_worker_get_count(); i < starpu_worker_get_count() + starpu_combined_worker_get_count(); i++)
	{
		if (starpu_bitmap_get(component->workers, i))
		{
			/* Component has this combined worker, check whether the
			 * context has all the corresponding workers */
			int worker_size;
			int *combined_workerid;
			starpu_combined_worker_get_description(i, &worker_size, &combined_workerid);
			for (j = 0; j < (unsigned) worker_size; j++)
				if (!starpu_bitmap_get(workers_in_ctx, combined_workerid[j]))
					goto nocombined;
			/* We have all workers, add it */
			starpu_bitmap_set(component->workers_in_ctx, i);
		}
nocombined:
		(void)0;
	}
	for(i = 0; i < component->nchildren; i++)
	{
		struct starpu_sched_component * child = component->children[i];
		_starpu_sched_component_update_workers_in_ctx(child, sched_ctx_id);
	}
	set_properties(component);
	component->notify_change_workers(component);
}



/******************************************************************************
 *          			Scheduling Trees' helper functions        			  *
 ******************************************************************************/



struct starpu_bitmap * _starpu_get_worker_mask(unsigned sched_ctx_id)
{
	STARPU_ASSERT(sched_ctx_id < STARPU_NMAX_SCHED_CTXS);
	struct starpu_sched_tree * t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	STARPU_ASSERT(t);
	return t->workers;
}

void starpu_sched_tree_update_workers_in_ctx(struct starpu_sched_tree * t)
{
	STARPU_ASSERT(t);
	if (t->root)
		_starpu_sched_component_update_workers_in_ctx(t->root, t->sched_ctx_id);
}

void starpu_sched_tree_update_workers(struct starpu_sched_tree * t)
{
	STARPU_ASSERT(t);
	if (t->root)
		_starpu_sched_component_update_workers(t->root);
}



/******************************************************************************
 *          			Scheduling Trees' Functions                			  *
 *  	Most of them are used to define the starpu_sched_policy interface     *
 ******************************************************************************/

void starpu_sched_component_connect(struct starpu_sched_component *parent, struct starpu_sched_component *child)
{
	parent->add_child(parent, child);
	child->add_parent(child, parent);
	_STARPU_TRACE_SCHED_COMPONENT_CONNECT(parent,child);
}

int starpu_sched_tree_push_task(struct starpu_task * task)
{
	STARPU_ASSERT(task);
	unsigned sched_ctx_id = task->sched_ctx;
	struct starpu_sched_tree *tree = starpu_sched_ctx_get_policy_data(sched_ctx_id);

	int ret_val = starpu_sched_component_push_task(NULL, tree->root,task);

	/* Modular schedulers are not supposed to refuse tasks */
	STARPU_ASSERT(!ret_val);

	return 0;
}

int starpu_sched_component_push_task(struct starpu_sched_component *from STARPU_ATTRIBUTE_UNUSED, struct starpu_sched_component *to, struct starpu_task *task)
{
	int pushback;
	pushback = to->push_task(to, task);
	if (!pushback)
		_STARPU_TRACE_SCHED_COMPONENT_PUSH(from, to, task);
	return pushback;
}

struct starpu_task * starpu_sched_tree_pop_task(unsigned sched_ctx)
{
	unsigned workerid = starpu_worker_get_id_check();
	struct starpu_sched_component * component = starpu_sched_component_worker_get(sched_ctx, workerid);

	/* _starpu_sched_component_lock_worker(workerid) is called by component->pull_task()
	 */
	struct starpu_task * task = starpu_sched_component_pull_task(component,NULL);
	return task;
}

struct starpu_task * starpu_sched_component_pull_task(struct starpu_sched_component *from, struct starpu_sched_component *to)
{
	struct starpu_task *task = from->pull_task(from, to);
	if (task)
		_STARPU_TRACE_SCHED_COMPONENT_PULL(from, to, task);
	return task;
}


/* Pump mechanic to get the task flow rolling. Takes tasks from component and send them to the child.
   To be used by components with only one child */
struct starpu_task* starpu_sched_component_pump_to(struct starpu_sched_component *component, struct starpu_sched_component *child, int* success)
{
	int ret = 0;

	struct starpu_task * task;

	while (1)
	{
		task = component->pull_task(component,child);
		if (!task)
			break;
		ret = starpu_sched_component_push_task(component,child,task);
		if (ret)
			break;
		if(success)
			* success = 1;
	}
	if(task && ret)
		/* Return the task which couldn't actually be pushed */
		return task;

	return NULL;

}

struct starpu_task* starpu_sched_component_pump_downstream(struct starpu_sched_component *component, int* success)
{
	STARPU_ASSERT(component->nchildren == 1);
	return starpu_sched_component_pump_to(component, component->children[0], success);
}

void starpu_sched_tree_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	STARPU_ASSERT(sched_ctx_id < STARPU_NMAX_SCHED_CTXS);
	STARPU_ASSERT(workerids);
	struct starpu_sched_tree * t = starpu_sched_ctx_get_policy_data(sched_ctx_id);

	STARPU_COMPONENT_MUTEX_LOCK(&t->lock);
	_starpu_sched_component_lock_all_workers();

	unsigned i;
	for(i = 0; i < nworkers; i++)
		starpu_bitmap_set(t->workers, workerids[i]);

	starpu_sched_tree_update_workers_in_ctx(t);

	_starpu_sched_component_unlock_all_workers();
	STARPU_COMPONENT_MUTEX_UNLOCK(&t->lock);
}

void starpu_sched_tree_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	STARPU_ASSERT(sched_ctx_id < STARPU_NMAX_SCHED_CTXS);
	STARPU_ASSERT(workerids);
	struct starpu_sched_tree * t = starpu_sched_ctx_get_policy_data(sched_ctx_id);

	STARPU_COMPONENT_MUTEX_LOCK(&t->lock);
	_starpu_sched_component_lock_all_workers();

	unsigned i;
	for(i = 0; i < nworkers; i++)
		starpu_bitmap_unset(t->workers, workerids[i]);

	starpu_sched_tree_update_workers_in_ctx(t);

	_starpu_sched_component_unlock_all_workers();
	STARPU_COMPONENT_MUTEX_UNLOCK(&t->lock);
}

static struct starpu_sched_tree *trees[STARPU_NMAX_SCHED_CTXS];

struct starpu_sched_tree * starpu_sched_tree_create(unsigned sched_ctx_id)
{
	STARPU_ASSERT(sched_ctx_id < STARPU_NMAX_SCHED_CTXS);
	STARPU_ASSERT(!trees[sched_ctx_id]);
	struct starpu_sched_tree *t;
	_STARPU_CALLOC(t, 1, sizeof(*t));
	t->sched_ctx_id = sched_ctx_id;
	t->workers = starpu_bitmap_create();
	STARPU_PTHREAD_MUTEX_INIT(&t->lock,NULL);
	trees[sched_ctx_id] = t;
	return t;
}

void starpu_sched_tree_destroy(struct starpu_sched_tree * tree)
{
	STARPU_ASSERT(tree);
	STARPU_ASSERT(trees[tree->sched_ctx_id] == tree);
	trees[tree->sched_ctx_id] = NULL;
	if(tree->root)
		starpu_sched_component_destroy_rec(tree->root);
	starpu_bitmap_destroy(tree->workers);
	STARPU_PTHREAD_MUTEX_DESTROY(&tree->lock);
	free(tree);
}

struct starpu_sched_tree * starpu_sched_tree_get(unsigned sched_ctx_id)
{
	return trees[sched_ctx_id];
}


/******************************************************************************
 *          Interface Functions for Generic Scheduling Components             *
 ******************************************************************************/



void starpu_sched_component_add_child(struct starpu_sched_component* component, struct starpu_sched_component * child)
{
	STARPU_ASSERT(component && child);
	STARPU_ASSERT(!starpu_sched_component_is_simple_worker(component));
	unsigned i;
	for(i = 0; i < component->nchildren; i++)
	{
		STARPU_ASSERT(component->children[i] != component);
		STARPU_ASSERT(component->children[i] != NULL);
	}

	_STARPU_REALLOC(component->children, sizeof(struct starpu_sched_component *) * (component->nchildren + 1));
	component->children[component->nchildren] = child;
	component->nchildren++;
}

static void starpu_sched_component_remove_child(struct starpu_sched_component * component, struct starpu_sched_component * child)
{
	STARPU_ASSERT(component && child);
	STARPU_ASSERT(!starpu_sched_component_is_simple_worker(component));
	unsigned pos;
	for(pos = 0; pos < component->nchildren; pos++)
		if(component->children[pos] == child)
			break;
	STARPU_ASSERT(pos != component->nchildren);
	component->children[pos] = component->children[--component->nchildren];
}

static void starpu_sched_component_add_parent(struct starpu_sched_component* component, struct starpu_sched_component * parent)
{
	STARPU_ASSERT(component && parent);
	unsigned i;
	for(i = 0; i < component->nparents; i++)
	{
		STARPU_ASSERT(component->parents[i] != component);
		STARPU_ASSERT(component->parents[i] != NULL);
	}

	_STARPU_REALLOC(component->parents, sizeof(struct starpu_sched_component *) * (component->nparents + 1));
	component->parents[component->nparents] = parent;
	component->nparents++;
}

static void starpu_sched_component_remove_parent(struct starpu_sched_component * component, struct starpu_sched_component * parent)
{
	STARPU_ASSERT(component && parent);
	unsigned pos;
	for(pos = 0; pos < component->nparents; pos++)
		if(component->parents[pos] == parent)
			break;
	STARPU_ASSERT(pos != component->nparents);
	component->parents[pos] = component->parents[--component->nparents];
}

/* default implementation for component->pull_task()
 * just perform a recursive call on parent
 */
struct starpu_task * starpu_sched_component_parents_pull_task(struct starpu_sched_component * component, struct starpu_sched_component * to STARPU_ATTRIBUTE_UNUSED)
{
	STARPU_ASSERT(component);
	struct starpu_task * task = NULL;
	unsigned i;
	for(i=0; i < component->nparents; i++)
	{
		if(component->parents[i] == NULL)
			continue;
		else
		{
			task = starpu_sched_component_pull_task(component->parents[i], component);
			if(task)
				break;
		}
	}
	return task;
}

/* The default implementation of the can_push function is a recursive call to its parents.
 * A personally-made can_push in a component (like in prio components) is necessary to catch
 * this recursive call somewhere, if the user wants to exploit it.
 */
int starpu_sched_component_can_push(struct starpu_sched_component * component, struct starpu_sched_component * to STARPU_ATTRIBUTE_UNUSED)
{
	STARPU_ASSERT(component);
	int ret = 0;
	if(component->nparents > 0)
	{
		unsigned i;
		for(i=0; i < component->nparents; i++)
		{
			struct starpu_sched_component * parent = component->parents[i];
			if(parent != NULL)
				ret = parent->can_push(parent, component);
			if(ret)
				break;
		}
	}
	return ret;
}

/* A can_pull call will try to wake up one worker associated to the childs of the
 * component. It is currenly called by components which holds a queue (like fifo and prio
 * components) to signify its childs that a task has been pushed on its local queue.
 */
int starpu_sched_component_can_pull(struct starpu_sched_component * component)
{
	STARPU_ASSERT(component);
	STARPU_ASSERT(!starpu_sched_component_is_worker(component));
	unsigned i;
	for(i = 0; i < component->nchildren; i++)
	{
		if (component->children[i]->can_pull(component->children[i]))
			return 1;
	}
	return 0;
}


/* A can_pull call will try to wake up one worker associated to the childs of the
 * component. It is currenly called by components which holds a queue (like fifo and prio
 * components) to signify its childs that a task has been pushed on its local queue.
 */
int starpu_sched_component_can_pull_all(struct starpu_sched_component * component)
{
	STARPU_ASSERT(component);
	STARPU_ASSERT(!starpu_sched_component_is_worker(component));
	unsigned i;
	for(i = 0; i < component->nchildren; i++)
		component->children[i]->can_pull(component->children[i]);
	return 0;
}


/* Alternative can_pull which says that this component does not want
   to pull but prefers that you push. It can be used by decision
   components, in which decisions are usually taken in their push()
   functions */
int starpu_sched_component_send_can_push_to_parents(struct starpu_sched_component * component)
{
	STARPU_ASSERT(component);
	STARPU_ASSERT(!starpu_sched_component_is_worker(component));

	unsigned i;
	int ret = 0;
	for(i=0; i < component->nparents; i++)
	{
		if(component->parents[i] == NULL)
			continue;
		else
		{
			ret = component->parents[i]->can_push(component->parents[i], component);
			if(ret)
				break;
		}
	}
	return ret != 0;
}


double starpu_sched_component_estimated_load(struct starpu_sched_component * component)
{
	double sum = 0.0;
	unsigned i;
	for( i = 0; i < component->nchildren; i++)
	{
		struct starpu_sched_component * c = component->children[i];
		sum += c->estimated_load(c);
	}
	return sum;
}

double starpu_sched_component_estimated_end_min_add(struct starpu_sched_component * component, double exp_len)
{
	STARPU_ASSERT(component);
	double min = DBL_MAX;
	unsigned i;
	double ends[component->nchildren];
	for(i = 0; i < component->nchildren; i++)
	{
		double tmp = ends[i] = component->children[i]->estimated_end(component->children[i]);
		if(tmp < min)
			min = tmp;
	}
	if (exp_len > 0)
	{
		/* We don't know which workers will do this, assume it will be
		 * evenly distributed to existing work */
		int card = starpu_bitmap_cardinal(component->workers_in_ctx);
		if (card == 0)
			/* Oops, no resources to compute our tasks. Let's just hope that
			 * we will be given one at some point */
			card = 1;
		for(i = 0; i < component->nchildren; i++)
		{
			exp_len += ends[i] - min;
		}
		min += exp_len / card;
	}
	return min;
}

double starpu_sched_component_estimated_end_min(struct starpu_sched_component * component)
{
  return starpu_sched_component_estimated_end_min_add(component, 0.);
}

double starpu_sched_component_estimated_end_average(struct starpu_sched_component * component)
{
	STARPU_ASSERT(component);
	double sum = 0.0;
	unsigned i;
	for(i = 0; i < component->nchildren; i++)
		sum += component->children[i]->estimated_end(component->children[i]);
	return sum / component->nchildren;
}

static void take_component_and_does_nothing(struct starpu_sched_component * component STARPU_ATTRIBUTE_UNUSED)
{
}

struct starpu_sched_component * starpu_sched_component_create(struct starpu_sched_tree *tree, const char *name)
{
	struct starpu_sched_component *component;
	_STARPU_CALLOC(component, 1, sizeof(*component));
	component->tree = tree;
	component->workers = starpu_bitmap_create();
	component->workers_in_ctx = starpu_bitmap_create();
	component->add_child = starpu_sched_component_add_child;
	component->remove_child = starpu_sched_component_remove_child;
	component->add_parent = starpu_sched_component_add_parent;
	component->remove_parent = starpu_sched_component_remove_parent;
	component->pull_task = starpu_sched_component_parents_pull_task;
	component->can_push = starpu_sched_component_can_push;
	component->can_pull = starpu_sched_component_can_pull;
	component->estimated_load = starpu_sched_component_estimated_load;
	component->estimated_end = starpu_sched_component_estimated_end_min;
	component->deinit_data = take_component_and_does_nothing;
	component->notify_change_workers = take_component_and_does_nothing;
	component->name = strdup(name);
	_STARPU_TRACE_SCHED_COMPONENT_NEW(component);
	return component;
}
