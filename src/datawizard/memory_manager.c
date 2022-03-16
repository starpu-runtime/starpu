/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>
#include <common/utils.h>
#include <common/thread.h>
#include <common/fxt.h>
#include <datawizard/memory_manager.h>
#include <datawizard/memory_nodes.h>
#include <core/workers.h>
#include <starpu_stdlib.h>

int _starpu_memory_manager_init()
{
	int i;

	for(i=0 ; i<STARPU_MAXNODES ; i++)
	{
		struct _starpu_node *node = _starpu_get_node_struct(i);
		node->global_size = 0;
		node->used_size = 0;
		/* This is accessed for statistics outside the lock, don't care
		 * about that */
		STARPU_HG_DISABLE_CHECKING(node->used_size);
		STARPU_HG_DISABLE_CHECKING(node->global_size);
		node->waiting_size = 0;
		STARPU_PTHREAD_MUTEX_INIT(&node->lock_nodes, NULL);
		STARPU_PTHREAD_COND_INIT(&node->cond_nodes, NULL);
	}
	return 0;
}

void _starpu_memory_manager_set_global_memory_size(unsigned node, size_t size)
{
	struct _starpu_node *node_struct = _starpu_get_node_struct(node);
	STARPU_PTHREAD_MUTEX_LOCK(&node_struct->lock_nodes);
	if (!node_struct->global_size)
	{
		node_struct->global_size = size;
		_STARPU_DEBUG("Global size for node %u is %ld\n", node, (long)node_struct->global_size);
	}
	else
	{
		STARPU_ASSERT(node_struct->global_size == size);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&node_struct->lock_nodes);
}

size_t _starpu_memory_manager_get_global_memory_size(unsigned node)
{
	return _starpu_get_node_struct(node)->global_size;
}


int starpu_memory_allocate(unsigned node, size_t size, int flags)
{
	struct _starpu_node *node_struct = _starpu_get_node_struct(node);
	int ret;

	STARPU_PTHREAD_MUTEX_LOCK(&node_struct->lock_nodes);
	if (flags & STARPU_MEMORY_WAIT)
	{
		struct _starpu_worker *worker = _starpu_get_local_worker_key();
		enum _starpu_worker_status old_status = STATUS_UNKNOWN;

		if (worker)
		{
			old_status = worker->status;
			if (!(old_status & STATUS_WAITING))
				_starpu_add_worker_status(worker, STATUS_INDEX_WAITING, NULL);
		}

		while (node_struct->used_size + size > node_struct->global_size)
		{
			/* Tell deallocators we need this amount */
			if (!node_struct->waiting_size || size < node_struct->waiting_size)
				node_struct->waiting_size = size;

			/* Wait for it */
			STARPU_PTHREAD_COND_WAIT(&node_struct->cond_nodes, &node_struct->lock_nodes);
		}

		if (worker)
		{
			if (!(old_status & STATUS_WAITING))
				_starpu_clear_worker_status(worker, STATUS_INDEX_WAITING, NULL);
		}

		/* And take it */
		node_struct->used_size += size;
		_STARPU_TRACE_USED_MEM(node, node_struct->used_size);
		ret = 0;
	}
	else if (flags & STARPU_MEMORY_OVERFLOW
			|| node_struct->global_size == 0
			|| node_struct->used_size + size <= node_struct->global_size)
	{
		node_struct->used_size += size;
		_STARPU_TRACE_USED_MEM(node, node_struct->used_size);
		ret = 0;
	}
	else
	{
		ret = -ENOMEM;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&node_struct->lock_nodes);
	return ret;
}

void starpu_memory_deallocate(unsigned node, size_t size)
{
	struct _starpu_node *node_struct = _starpu_get_node_struct(node);
	STARPU_PTHREAD_MUTEX_LOCK(&node_struct->lock_nodes);

	node_struct->used_size -= size;
	_STARPU_TRACE_USED_MEM(node, node_struct->used_size);

	/* If there's now room for waiters, wake them */
	if (node_struct->waiting_size &&
		node_struct->global_size - node_struct->used_size >= node_struct->waiting_size)
	{
		/* And have those not happy enough tell us the size again */
		node_struct->waiting_size = 0;
		STARPU_PTHREAD_COND_BROADCAST(&node_struct->cond_nodes);
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&node_struct->lock_nodes);
}

starpu_ssize_t starpu_memory_get_total(unsigned node)
{
	size_t size = _starpu_get_node_struct(node)->global_size;
	if (size == 0)
		return -1;
	else
		return size;
}

starpu_ssize_t starpu_memory_get_total_all_nodes()
{
	unsigned memnodes, i;
	memnodes = starpu_memory_nodes_get_count();
	starpu_ssize_t total = 0;
	for(i=0 ; i<memnodes ; i++)
	{
		starpu_ssize_t node = starpu_memory_get_total(i);
		if (node != -1)
			total += node;
	}
	return total;
}

starpu_ssize_t starpu_memory_get_available(unsigned node)
{
	starpu_ssize_t ret;
	size_t size = _starpu_get_node_struct(node)->global_size;
	if (size == 0)
		return -1;

	ret = size - _starpu_get_node_struct(node)->used_size;
	return ret;
}

starpu_ssize_t starpu_memory_get_available_all_nodes()
{
	unsigned memnodes, i;
	memnodes = starpu_memory_nodes_get_count();
	starpu_ssize_t avail = 0;
	for(i=0 ; i<memnodes ; i++)
	{
		starpu_ssize_t node = starpu_memory_get_available(i);
		if (node != -1)
			avail += node;
	}
	return avail;
}

size_t starpu_memory_get_used(unsigned node)
{
	return _starpu_get_node_struct(node)->used_size;
}

size_t starpu_memory_get_used_all_nodes()
{
	unsigned memnodes, i;
	memnodes = starpu_memory_nodes_get_count();
	size_t used = 0;
	for(i=0 ; i<memnodes ; i++)
	{
		size_t node = starpu_memory_get_used(i);
		used += node;
	}
	return used;
}

void starpu_memory_wait_available(unsigned node, size_t size)
{
	struct _starpu_node *node_struct = _starpu_get_node_struct(node);
	STARPU_PTHREAD_MUTEX_LOCK(&node_struct->lock_nodes);
	while (node_struct->used_size + size > node_struct->global_size)
	{
		/* Tell deallocators we need this amount */
		if (!node_struct->waiting_size || size < node_struct->waiting_size)
			node_struct->waiting_size = size;

		/* Wait for it */
		STARPU_PTHREAD_COND_WAIT(&node_struct->cond_nodes, &node_struct->lock_nodes);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&node_struct->lock_nodes);
}

int _starpu_memory_manager_test_allocate_size(unsigned node, size_t size)
{
	struct _starpu_node *node_struct = _starpu_get_node_struct(node);
	int ret;

	if (node_struct->global_size == 0)
		ret = 1;
	else if (node_struct->used_size + size <= node_struct->global_size)
		ret = 1;
	else
		ret = 0;
	return ret;
}
