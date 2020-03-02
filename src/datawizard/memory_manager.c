/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <core/workers.h>
#include <starpu_stdlib.h>

static size_t global_size[STARPU_MAXNODES];
static size_t used_size[STARPU_MAXNODES];

/* This is used as an optimization to avoid to wake up allocating threads for
 * each and every deallocation, only to find that there is still not enough
 * room.  */
/* Minimum amount being waited for */
static size_t waiting_size[STARPU_MAXNODES];

static starpu_pthread_mutex_t lock_nodes[STARPU_MAXNODES];
static starpu_pthread_cond_t cond_nodes[STARPU_MAXNODES];

int _starpu_memory_manager_init()
{
	int i;

	for(i=0 ; i<STARPU_MAXNODES ; i++)
	{
		global_size[i] = 0;
		used_size[i] = 0;
		/* This is accessed for statistics outside the lock, don't care
		 * about that */
		STARPU_HG_DISABLE_CHECKING(used_size[i]);
		STARPU_HG_DISABLE_CHECKING(global_size[i]);
		waiting_size[i] = 0;
		STARPU_PTHREAD_MUTEX_INIT(&lock_nodes[i], NULL);
		STARPU_PTHREAD_COND_INIT(&cond_nodes[i], NULL);
	}
	return 0;
}

void _starpu_memory_manager_set_global_memory_size(unsigned node, size_t size)
{
	STARPU_PTHREAD_MUTEX_LOCK(&lock_nodes[node]);
	if (!global_size[node])
	{
		global_size[node] = size;
		_STARPU_DEBUG("Global size for node %u is %ld\n", node, (long)global_size[node]);
	}
	else
	{
		STARPU_ASSERT(global_size[node] == size);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&lock_nodes[node]);
}

size_t _starpu_memory_manager_get_global_memory_size(unsigned node)
{
	return global_size[node];
}


int starpu_memory_allocate(unsigned node, size_t size, int flags)
{
	int ret;

	STARPU_PTHREAD_MUTEX_LOCK(&lock_nodes[node]);
	if (flags & STARPU_MEMORY_WAIT)
	{
		struct _starpu_worker *worker = _starpu_get_local_worker_key();
		enum _starpu_worker_status old_status = STATUS_UNKNOWN;

		if (worker)
		{
			old_status = worker->status;
			_starpu_set_worker_status(worker, STATUS_WAITING);
		}

		while (used_size[node] + size > global_size[node])
		{
			/* Tell deallocators we need this amount */
			if (!waiting_size[node] || size < waiting_size[node])
				waiting_size[node] = size;

			/* Wait for it */
			STARPU_PTHREAD_COND_WAIT(&cond_nodes[node], &lock_nodes[node]);
		}

		if (worker)
		{
			_starpu_set_worker_status(worker, old_status);
		}

		/* And take it */
		used_size[node] += size;
		_STARPU_TRACE_USED_MEM(node, used_size[node]);
		ret = 0;
	}
	else if (flags & STARPU_MEMORY_OVERFLOW
			|| global_size[node] == 0
			|| used_size[node] + size <= global_size[node])
	{
		used_size[node] += size;
		_STARPU_TRACE_USED_MEM(node, used_size[node]);
		ret = 0;
	}
	else
	{
		ret = -ENOMEM;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&lock_nodes[node]);
	return ret;
}

void starpu_memory_deallocate(unsigned node, size_t size)
{
	STARPU_PTHREAD_MUTEX_LOCK(&lock_nodes[node]);

	used_size[node] -= size;
	_STARPU_TRACE_USED_MEM(node, used_size[node]);

	/* If there's now room for waiters, wake them */
	if (waiting_size[node] &&
		global_size[node] - used_size[node] >= waiting_size[node])
	{
		/* And have those not happy enough tell us the size again */
		waiting_size[node] = 0;
		STARPU_PTHREAD_COND_BROADCAST(&cond_nodes[node]);
	}

	STARPU_PTHREAD_MUTEX_UNLOCK(&lock_nodes[node]);
}

starpu_ssize_t starpu_memory_get_total(unsigned node)
{
	if (global_size[node] == 0)
		return -1;
	else
		return global_size[node];
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
	if (global_size[node] == 0)
		return -1;

	ret = global_size[node] - used_size[node];
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

void starpu_memory_wait_available(unsigned node, size_t size)
{
	STARPU_PTHREAD_MUTEX_LOCK(&lock_nodes[node]);
	while (used_size[node] + size > global_size[node])
	{
		/* Tell deallocators we need this amount */
		if (!waiting_size[node] || size < waiting_size[node])
			waiting_size[node] = size;

		/* Wait for it */
		STARPU_PTHREAD_COND_WAIT(&cond_nodes[node], &lock_nodes[node]);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&lock_nodes[node]);
}

int _starpu_memory_manager_test_allocate_size(unsigned node, size_t size)
{
	int ret;

	if (global_size[node] == 0)
		ret = 1;
	else if (used_size[node] + size <= global_size[node])
		ret = 1;
	else
		ret = 0;
	return ret;
}
