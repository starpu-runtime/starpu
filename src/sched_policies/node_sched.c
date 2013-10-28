/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013 Simon Archipoff
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
#include <starpu_sched_node.h>
#include <starpu_thread_util.h>

#include <float.h>

#include "sched_node.h"


/* wake up worker workerid
 * if called by a worker it dont try to wake up himself
 */
static void wake_simple_worker(int workerid)
{
	STARPU_ASSERT(0 <= workerid && (unsigned)  workerid < starpu_worker_get_count());
	starpu_pthread_mutex_t * sched_mutex;
	starpu_pthread_cond_t * sched_cond;
	if(workerid == starpu_worker_get_id())
		return;
	starpu_worker_get_sched_condition(workerid, &sched_mutex, &sched_cond);
	STARPU_PTHREAD_MUTEX_LOCK(sched_mutex);
	STARPU_PTHREAD_COND_SIGNAL(sched_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(sched_mutex);
}

/* wake up all workers of a combined workers
 * this function must not be called during a pop (however this should not
 * even be possible) or you will have a dead lock
 */
static void wake_combined_worker(int workerid)
{
	STARPU_ASSERT( 0 <= workerid
		       && starpu_worker_get_count() <= (unsigned) workerid
		       && (unsigned) workerid < starpu_worker_get_count() + starpu_combined_worker_get_count());
	struct _starpu_combined_worker * combined_worker = _starpu_get_combined_worker_struct(workerid);
	int * list = combined_worker->combined_workerid;
	int size = combined_worker->worker_size;
	int i;
	for(i = 0; i < size; i++)
		wake_simple_worker(list[i]);
}


/* this function must not be called on worker nodes :
 * because this wouldn't have sense
 * and should dead lock
 */
void starpu_sched_node_available(struct starpu_sched_node * node)
{
	(void)node;
	STARPU_ASSERT(node);
	STARPU_ASSERT(!starpu_sched_node_is_worker(node));
#ifndef STARPU_NON_BLOCKING_DRIVERS
	int i;
	for(i = starpu_bitmap_first(node->workers_in_ctx);
	    i != -1;
	    i = starpu_bitmap_next(node->workers_in_ctx, i))
	{
		if(i < (int) starpu_worker_get_count())
			wake_simple_worker(i);
		else
			wake_combined_worker(i);
	}
#endif
}

/* default implementation for node->pop_task()
 * just perform a recursive call on father
 */
static struct starpu_task * pop_task_node(struct starpu_sched_node * node, unsigned sched_ctx_id)
{
	STARPU_ASSERT(sched_ctx_id < STARPU_NMAX_SCHED_CTXS);
	STARPU_ASSERT(node);
	if(node->fathers[sched_ctx_id] == NULL)
		return NULL;
	else
		return node->fathers[sched_ctx_id]->pop_task(node->fathers[sched_ctx_id], sched_ctx_id);
}


void starpu_sched_node_set_father(struct starpu_sched_node *node,
				  struct starpu_sched_node *father_node,
				  unsigned sched_ctx_id)
{
	STARPU_ASSERT(sched_ctx_id < STARPU_NMAX_SCHED_CTXS);
	STARPU_ASSERT(node);
	node->fathers[sched_ctx_id] = father_node;
}



/******************************************************************************
 *          functions for struct starpu_sched_policy interface                *
 ******************************************************************************/
int starpu_sched_tree_push_task(struct starpu_task * task)
{
	STARPU_ASSERT(task);
	unsigned sched_ctx_id = task->sched_ctx;
	struct starpu_sched_tree *tree = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	int workerid = starpu_worker_get_id();
	/* application should take tree->lock to prevent concurent acces from hypervisor
	 * worker take they own mutexes
	 */
	if(-1 == workerid)
		STARPU_PTHREAD_MUTEX_LOCK(&tree->lock);
	else
		_starpu_sched_node_lock_worker(workerid);
		
	int ret_val = tree->root->push_task(tree->root,task);
	if(-1 == workerid)
		STARPU_PTHREAD_MUTEX_UNLOCK(&tree->lock);
	else
		_starpu_sched_node_unlock_worker(workerid);
	return ret_val;
}

struct starpu_task * starpu_sched_tree_pop_task(unsigned sched_ctx_id)
{
	STARPU_ASSERT(sched_ctx_id < STARPU_NMAX_SCHED_CTXS);
	int workerid = starpu_worker_get_id();
	struct starpu_sched_node * node = starpu_sched_node_worker_get(workerid);

	/* _starpu_sched_node_lock_worker(workerid) is called by node->pop_task()
	 */
	struct starpu_task * task = node->pop_task(node, sched_ctx_id);
	return task;
}

void starpu_sched_tree_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	STARPU_ASSERT(sched_ctx_id < STARPU_NMAX_SCHED_CTXS);
	STARPU_ASSERT(workerids);
	struct starpu_sched_tree * t = starpu_sched_ctx_get_policy_data(sched_ctx_id);

	STARPU_PTHREAD_MUTEX_LOCK(&t->lock);
	_starpu_sched_node_lock_all_workers();

	unsigned i;
	for(i = 0; i < nworkers; i++)
		starpu_bitmap_set(t->workers, workerids[i]);

	starpu_sched_tree_update_workers_in_ctx(t);

	_starpu_sched_node_unlock_all_workers();
	STARPU_PTHREAD_MUTEX_UNLOCK(&t->lock);
}

void starpu_sched_tree_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers)
{
	STARPU_ASSERT(sched_ctx_id < STARPU_NMAX_SCHED_CTXS);
	STARPU_ASSERT(workerids);
	struct starpu_sched_tree * t = starpu_sched_ctx_get_policy_data(sched_ctx_id);

	STARPU_PTHREAD_MUTEX_LOCK(&t->lock);
	_starpu_sched_node_lock_all_workers();

	unsigned i;
	for(i = 0; i < nworkers; i++)
		starpu_bitmap_unset(t->workers, workerids[i]);

	starpu_sched_tree_update_workers_in_ctx(t);

	_starpu_sched_node_unlock_all_workers();
	STARPU_PTHREAD_MUTEX_UNLOCK(&t->lock);
}




void starpu_sched_node_destroy_rec(struct starpu_sched_node * node, unsigned sched_ctx_id)
{
	if(node == NULL)
		return;
	struct starpu_sched_node ** stack = NULL;
	int top = -1;
#define PUSH(n)								\
	do{								\
		stack = realloc(stack, sizeof(*stack) * (top + 2));	\
		stack[++top] = n;					\
	}while(0)
#define POP() stack[top--]
#define EMPTY() (top == -1)

	/* we want to delete all subtrees exept if a pointer in fathers point in an other tree
	 * ie an other context
	 */
	node->fathers[sched_ctx_id] = NULL;
	int shared = 0;
	{
		int i;
		for(i = 0; i < STARPU_NMAX_SCHED_CTXS; i++)
			if(node->fathers[i] != NULL)
				shared = 1;
	}
	if(!shared)
		PUSH(node);
	while(!EMPTY())
	{
		struct starpu_sched_node * n = POP();
		int i;
		for(i = 0; i < n->nchilds; i++)
		{
			struct starpu_sched_node * child = n->childs[i];
			int j;
			shared = 0;
			STARPU_ASSERT(child->fathers[sched_ctx_id] == n);
			child->fathers[sched_ctx_id] = NULL;
			for(j = 0; j < STARPU_NMAX_SCHED_CTXS; j++)
			{
				if(child->fathers[j] != NULL)/* child is shared */
					shared = 1;
			}
			if(!shared)/* if not shared we want to destroy it and his childs */
				PUSH(child);
		}
		starpu_sched_node_destroy(n);
	}
	free(stack);
}

struct starpu_sched_tree * starpu_sched_tree_create(unsigned sched_ctx_id)
{
	STARPU_ASSERT(sched_ctx_id < STARPU_NMAX_SCHED_CTXS);
	struct starpu_sched_tree * t = malloc(sizeof(*t));
	memset(t, 0, sizeof(*t));
	t->sched_ctx_id = sched_ctx_id;
	t->workers = starpu_bitmap_create();
	STARPU_PTHREAD_MUTEX_INIT(&t->lock,NULL);
	return t;
}

void starpu_sched_tree_destroy(struct starpu_sched_tree * tree)
{
	STARPU_ASSERT(tree);
	if(tree->root)
		starpu_sched_node_destroy_rec(tree->root, tree->sched_ctx_id);
	starpu_bitmap_destroy(tree->workers);
	STARPU_PTHREAD_MUTEX_DESTROY(&tree->lock);
	free(tree);
}
void starpu_sched_node_add_child(struct starpu_sched_node* node, struct starpu_sched_node * child)
{
	STARPU_ASSERT(node && child);
	STARPU_ASSERT(!starpu_sched_node_is_worker(node));
	int i;
	for(i = 0; i < node->nchilds; i++){
		STARPU_ASSERT(node->childs[i] != node);
		STARPU_ASSERT(node->childs[i] != NULL);
	}

	node->childs = realloc(node->childs, sizeof(struct starpu_sched_node *) * (node->nchilds + 1));
	node->childs[node->nchilds] = child;
	node->nchilds++;
}
void starpu_sched_node_remove_child(struct starpu_sched_node * node, struct starpu_sched_node * child)
{
	STARPU_ASSERT(node && child);
	STARPU_ASSERT(!starpu_sched_node_is_worker(node));
	int pos;
	for(pos = 0; pos < node->nchilds; pos++)
		if(node->childs[pos] == child)
			break;
	STARPU_ASSERT(pos != node->nchilds);
	node->childs[pos] = node->childs[--node->nchilds];
}

struct starpu_bitmap * _starpu_get_worker_mask(unsigned sched_ctx_id)
{
	STARPU_ASSERT(sched_ctx_id < STARPU_NMAX_SCHED_CTXS);
	struct starpu_sched_tree * t = starpu_sched_ctx_get_policy_data(sched_ctx_id);
	STARPU_ASSERT(t);
	return t->workers;
}

static double estimated_load(struct starpu_sched_node * node)
{
	double sum = 0.0;
	int i;
	for( i = 0; i < node->nchilds; i++)
	{
		struct starpu_sched_node * c = node->childs[i];
		sum += c->estimated_load(c);
	}
	return sum;
}

static double _starpu_sched_node_estimated_end_min(struct starpu_sched_node * node)
{
	STARPU_ASSERT(node);
	double min = DBL_MAX;
	int i;
	for(i = 0; i < node->nchilds; i++)
	{
		double tmp = node->childs[i]->estimated_end(node->childs[i]);
		if(tmp < min)
			min = tmp;
	}
	return min;
}

/* this function find the best implementation or an implementation that need to be calibrated for a worker available
 * and set prediction in *length. nan if a implementation need to be calibrated, 0.0 if no perf model are available
 * return false if no worker on the node can execute that task
 */
int STARPU_WARN_UNUSED_RESULT starpu_sched_node_execute_preds(struct starpu_sched_node * node, struct starpu_task * task, double * length)
{
	STARPU_ASSERT(node && task);
	int can_execute = 0;
	starpu_task_bundle_t bundle = task->bundle;
	double len = DBL_MAX;
	

	int workerid;
	for(workerid = starpu_bitmap_first(node->workers_in_ctx);
	    workerid != -1;
	    workerid = starpu_bitmap_next(node->workers_in_ctx, workerid))
	{
		struct starpu_perfmodel_arch* archtype = starpu_worker_get_perf_archtype(workerid);
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
				if(_STARPU_IS_ZERO(d) && !can_execute)
				{
					can_execute = 1;
					continue;
				}
				if(d < len)
				{
					len = d;
				}
			}
		}
		if(STARPU_SCHED_NODE_IS_HOMOGENEOUS(node))
			break;
	}

	if(len == DBL_MAX) /* we dont have perf model */
		len = 0.0; 
	if(length)
		*length = len;
	return can_execute;
}

/* very similar function that dont compute prediction */
int starpu_sched_node_can_execute_task(struct starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_ASSERT(task);
	STARPU_ASSERT(node);
	unsigned nimpl;
	int worker;
	for (nimpl = 0; nimpl < STARPU_MAXIMPLEMENTATIONS; nimpl++)
		for(worker = starpu_bitmap_first(node->workers_in_ctx);
		    -1 != worker;
		    worker = starpu_bitmap_next(node->workers_in_ctx, worker))
			if (starpu_worker_can_execute_task(worker, task, nimpl)
			     || starpu_combined_worker_can_execute_task(worker, task, nimpl))
			    return 1;
	return 0;
}

/* compute the average of transfer length for tasks on all workers
 * maybe this should be optimised if all workers are under the same numa node
 */
double starpu_sched_node_transfer_length(struct starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_ASSERT(node && task);
	int nworkers = starpu_bitmap_cardinal(node->workers_in_ctx);
	double sum = 0.0;
	int worker;
	if(STARPU_SCHED_NODE_IS_SINGLE_MEMORY_NODE(node))
	{
		unsigned memory_node  = starpu_worker_get_memory_node(starpu_bitmap_first(node->workers_in_ctx));
		if(task->bundle)
			return starpu_task_bundle_expected_data_transfer_time(task->bundle,memory_node);
		else
			return starpu_task_expected_data_transfer_time(memory_node, task);
	}

	for(worker = starpu_bitmap_first(node->workers_in_ctx);
	    worker != -1;
	    worker = starpu_bitmap_next(node->workers_in_ctx, worker))
	{
		unsigned memory_node  = starpu_worker_get_memory_node(worker);
		if(task->bundle)
		{
			sum += starpu_task_bundle_expected_data_transfer_time(task->bundle,memory_node);
		}
		else
		{
			sum += starpu_task_expected_data_transfer_time(memory_node, task);
			/* sum += starpu_task_expected_conversion_time(task, starpu_worker_get_perf_archtype(worker), impl ?)
			 * I dont know what to do as we dont know what implementation would be used here...
			 */
		}
	}
	return sum / nworkers;
}

void starpu_sched_node_prefetch_on_node(struct starpu_sched_node * node, struct starpu_task * task)
{
       if (starpu_get_prefetch_flag() && (node->properties >= STARPU_SCHED_NODE_SINGLE_MEMORY_NODE))
       {
               int worker = starpu_bitmap_first(node->workers_in_ctx);
               unsigned memory_node = starpu_worker_get_memory_node(worker);
               starpu_prefetch_task_input_on_node(task, memory_node);
       }
}


void take_node_and_does_nothing(struct starpu_sched_node * node STARPU_ATTRIBUTE_UNUSED)
{
}

struct starpu_sched_node * starpu_sched_node_create(void)
{
	struct starpu_sched_node * node = malloc(sizeof(*node));
	memset(node,0,sizeof(*node));
	node->workers = starpu_bitmap_create();
	node->workers_in_ctx = starpu_bitmap_create();
	node->add_child = starpu_sched_node_add_child;
	node->remove_child = starpu_sched_node_remove_child;
	node->pop_task = pop_task_node;
	node->estimated_load = estimated_load;
	node->estimated_end = _starpu_sched_node_estimated_end_min;
	node->deinit_data = take_node_and_does_nothing;
	node->notify_change_workers = take_node_and_does_nothing;
	return node;
}

/* remove all child
 * for all child of node, if child->fathers[x] == node, set child->fathers[x] to null 
 * call node->deinit_data
 */
void starpu_sched_node_destroy(struct starpu_sched_node *node)
{
	STARPU_ASSERT(node);
	if(starpu_sched_node_is_worker(node))
		return;
	int i,j;
	for(i = 0; i < node->nchilds; i++)
	{
		struct starpu_sched_node * child = node->childs[i];
		for(j = 0; j < STARPU_NMAX_SCHED_CTXS; j++)
			if(child->fathers[i] == node)
				child->fathers[i] = NULL;

	}
	while(node->nchilds != 0)
		node->remove_child(node, node->childs[0]);
	node->deinit_data(node);
	free(node->childs);
	starpu_bitmap_destroy(node->workers);
	starpu_bitmap_destroy(node->workers_in_ctx);
	free(node);
}

static void set_properties(struct starpu_sched_node * node)
{
	STARPU_ASSERT(node);
	node->properties = 0;
	STARPU_ASSERT(starpu_bitmap_cardinal(node->workers_in_ctx) > 0);

	int worker = starpu_bitmap_first(node->workers_in_ctx);
	uint32_t first_worker = _starpu_get_worker_struct(worker)->worker_mask;
	unsigned first_memory_node = _starpu_get_worker_struct(worker)->memory_node;
	int is_homogeneous = 1;
	int is_all_same_node = 1;
	for(;
	    worker != -1;
	    worker = starpu_bitmap_next(node->workers_in_ctx, worker))		
	{
		if(first_worker != _starpu_get_worker_struct(worker)->worker_mask)
			is_homogeneous = 0;
		if(first_memory_node != _starpu_get_worker_struct(worker)->memory_node)
			is_all_same_node = 0;
	}
	

	if(is_homogeneous)
		node->properties |= STARPU_SCHED_NODE_HOMOGENEOUS;
	if(is_all_same_node)
		node->properties |= STARPU_SCHED_NODE_SINGLE_MEMORY_NODE;
}


/* recursively set the node->workers member of node's subtree
 */
void _starpu_sched_node_update_workers(struct starpu_sched_node * node)
{
	STARPU_ASSERT(node);
	if(starpu_sched_node_is_worker(node))
		return;
	starpu_bitmap_unset_all(node->workers);
	int i;
	for(i = 0; i < node->nchilds; i++)
	{
		_starpu_sched_node_update_workers(node->childs[i]);
		starpu_bitmap_or(node->workers, node->childs[i]->workers);
		node->notify_change_workers(node);
	}
}

/* recursively set the node->workers_in_ctx in node's subtree
 */
void _starpu_sched_node_update_workers_in_ctx(struct starpu_sched_node * node, unsigned sched_ctx_id)
{
	STARPU_ASSERT(node);
	if(starpu_sched_node_is_worker(node))
		return;
	struct starpu_bitmap * workers_in_ctx = _starpu_get_worker_mask(sched_ctx_id);
	starpu_bitmap_unset_and(node->workers_in_ctx,node->workers, workers_in_ctx);
	int i,j;
	for(i = 0; i < node->nchilds; i++)
	{
		struct starpu_sched_node * child = node->childs[i];
		_starpu_sched_node_update_workers_in_ctx(child, sched_ctx_id);
		for(j = 0; j < STARPU_NMAX_SCHED_CTXS; j++)
			if(child->fathers[j] == node)
			{
				starpu_bitmap_or(node->workers_in_ctx, child->workers_in_ctx);
				break;
			}
	}
	set_properties(node);
	node->notify_change_workers(node);
}

void starpu_sched_tree_update_workers_in_ctx(struct starpu_sched_tree * t)
{
	STARPU_ASSERT(t);
	_starpu_sched_node_update_workers_in_ctx(t->root, t->sched_ctx_id);
}

void starpu_sched_tree_update_workers(struct starpu_sched_tree * t)
{
	STARPU_ASSERT(t);
	_starpu_sched_node_update_workers(t->root);
}
