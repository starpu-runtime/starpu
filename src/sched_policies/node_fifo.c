/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013  Simon Archipoff
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

#include <starpu_sched_node.h>
#include <starpu_scheduler.h>

#include "prio_deque.h"


struct _starpu_fifo_data
{
	struct _starpu_prio_deque fifo;
	starpu_pthread_mutex_t mutex;
};

void fifo_node_deinit_data(struct starpu_sched_node * node)
{
	STARPU_ASSERT(node && node->data);
	struct _starpu_fifo_data * f = node->data;
	_starpu_prio_deque_destroy(&f->fifo);
	STARPU_PTHREAD_MUTEX_DESTROY(&f->mutex);
	free(f);
}

static double fifo_estimated_end(struct starpu_sched_node * node)
{
	STARPU_ASSERT(node && node->data);
	struct _starpu_fifo_data * data = node->data;
	struct _starpu_prio_deque * fifo = &data->fifo;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	int card = starpu_bitmap_cardinal(node->workers_in_ctx);
	STARPU_ASSERT(card != 0);
	STARPU_PTHREAD_MUTEX_LOCK(mutex);
	fifo->exp_start = STARPU_MAX(fifo->exp_start, starpu_timing_now());
	double estimated_end = fifo->exp_start + fifo->exp_len / card;
	STARPU_PTHREAD_MUTEX_UNLOCK(mutex);

	return estimated_end;
}

static double fifo_estimated_load(struct starpu_sched_node * node)
{
	STARPU_ASSERT(node && node->data);
	STARPU_ASSERT(starpu_bitmap_cardinal(node->workers_in_ctx) != 0);
	struct _starpu_fifo_data * data = node->data;
	struct _starpu_prio_deque * fifo = &data->fifo;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	double relative_speedup = 0.0;
	double load;
	if(STARPU_SCHED_NODE_IS_HOMOGENEOUS(node))
	{		
		int first_worker = starpu_bitmap_first(node->workers_in_ctx);
		relative_speedup = starpu_worker_get_relative_speedup(starpu_worker_get_perf_archtype(first_worker));
		STARPU_PTHREAD_MUTEX_LOCK(mutex);
		load = fifo->ntasks / relative_speedup;
		STARPU_PTHREAD_MUTEX_UNLOCK(mutex);
		return load;
	}
	else
	{
		int i;
		for(i = starpu_bitmap_first(node->workers_in_ctx);
		    i != -1;
		    i = starpu_bitmap_next(node->workers_in_ctx, i))
			relative_speedup += starpu_worker_get_relative_speedup(starpu_worker_get_perf_archtype(i));
		relative_speedup /= starpu_bitmap_cardinal(node->workers_in_ctx);
		STARPU_ASSERT(!_STARPU_IS_ZERO(relative_speedup));
		STARPU_PTHREAD_MUTEX_LOCK(mutex);
		load = fifo->ntasks / relative_speedup;
		STARPU_PTHREAD_MUTEX_UNLOCK(mutex);
	}
	int i;
	for(i = 0; i < node->nchilds; i++)
	{
		struct starpu_sched_node * c = node->childs[i];
		load += c->estimated_load(c);
	}
	return load;
}

static int fifo_push_task(struct starpu_sched_node * node, struct starpu_task * task)
{
	STARPU_ASSERT(node && node->data && task);
	STARPU_ASSERT(starpu_sched_node_can_execute_task(node,task));
	struct _starpu_fifo_data * data = node->data;
	struct _starpu_prio_deque * fifo = &data->fifo;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	STARPU_PTHREAD_MUTEX_LOCK(mutex);
	int ret = _starpu_prio_deque_push_task(fifo, task);
	if(!isnan(task->predicted))
	{
		fifo->exp_len += task->predicted;
		fifo->exp_end = fifo->exp_start + fifo->exp_len;
	}
	STARPU_ASSERT(!isnan(fifo->exp_end));
	STARPU_ASSERT(!isnan(fifo->exp_len));
	STARPU_ASSERT(!isnan(fifo->exp_start));
	STARPU_PTHREAD_MUTEX_UNLOCK(mutex);

	starpu_sched_node_available(node);

	return ret;
}

static struct starpu_task * fifo_pop_task(struct starpu_sched_node * node, unsigned sched_ctx_id)
{
	STARPU_ASSERT(node && node->data);
	struct _starpu_fifo_data * data = node->data;
	struct _starpu_prio_deque * fifo = &data->fifo;
	starpu_pthread_mutex_t * mutex = &data->mutex;
	STARPU_PTHREAD_MUTEX_LOCK(mutex);
	struct starpu_task * task = _starpu_prio_deque_pop_task_for_worker(fifo, starpu_worker_get_id());
	if(task)
	{

		if(!isnan(task->predicted))
		{
			fifo->exp_start = starpu_timing_now() + task->predicted;
			fifo->exp_len -= task->predicted;
		}
		fifo->exp_end = fifo->exp_start + fifo->exp_len;
		if(fifo->ntasks == 0)
			fifo->exp_len = 0.0;
	}
	STARPU_ASSERT(!isnan(fifo->exp_end));
	STARPU_ASSERT(!isnan(fifo->exp_len));
	STARPU_ASSERT(!isnan(fifo->exp_start));
	STARPU_PTHREAD_MUTEX_UNLOCK(mutex);
	if(task)
		return task;
	struct starpu_sched_node * father = node->fathers[sched_ctx_id];
	if(father)
		return father->pop_task(father,sched_ctx_id);
	return NULL;
}

int starpu_sched_node_is_fifo(struct starpu_sched_node * node)
{
	return node->push_task == fifo_push_task;
}

struct starpu_sched_node * starpu_sched_node_fifo_create(void * arg STARPU_ATTRIBUTE_UNUSED)
{
	struct starpu_sched_node * node = starpu_sched_node_create();
	struct _starpu_fifo_data * data = malloc(sizeof(*data));
	_starpu_prio_deque_init(&data->fifo);
	STARPU_PTHREAD_MUTEX_INIT(&data->mutex,NULL);
	node->data = data;
	node->estimated_end = fifo_estimated_end;
	node->estimated_load = fifo_estimated_load;
	node->push_task = fifo_push_task;
	node->pop_task = fifo_pop_task;
	node->deinit_data = fifo_node_deinit_data;
	return node;
}




static void initialize_eager_center_policy(unsigned sched_ctx_id)
{
	starpu_sched_ctx_create_worker_collection(sched_ctx_id, STARPU_WORKER_LIST);
	struct starpu_sched_tree *t = starpu_sched_tree_create(sched_ctx_id);
 	t->root = starpu_sched_node_fifo_create(NULL);
	unsigned i;
	for(i = 0; i < starpu_worker_get_count() + starpu_combined_worker_get_count(); i++)
	{
		struct starpu_sched_node * node = starpu_sched_node_worker_get(i);
		if(!node)
			continue;
		node->fathers[sched_ctx_id] = t->root;
		t->root->add_child(t->root, node);
	}
	starpu_sched_tree_update_workers(t);
	starpu_sched_ctx_set_policy_data(sched_ctx_id, (void*)t);
}

static void deinitialize_eager_center_policy(unsigned sched_ctx_id)
{
	struct starpu_sched_tree *tree = (struct starpu_sched_tree*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	starpu_sched_tree_destroy(tree);
	starpu_sched_ctx_delete_worker_collection(sched_ctx_id);
}

struct starpu_sched_policy _starpu_sched_tree_eager_policy =
{
	.init_sched = initialize_eager_center_policy,
	.deinit_sched = deinitialize_eager_center_policy,
	.add_workers = starpu_sched_tree_add_workers,
	.remove_workers = starpu_sched_tree_remove_workers,
	.push_task = starpu_sched_tree_push_task,
	.pop_task = starpu_sched_tree_pop_task,
	.pre_exec_hook = starpu_sched_node_worker_pre_exec_hook,
	.post_exec_hook = starpu_sched_node_worker_post_exec_hook,
	.pop_every_task = NULL,//pop_every_task_eager_policy,
	.policy_name = "tree",
	.policy_description = "test tree policy"
};
