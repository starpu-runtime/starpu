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
#include <datawizard/memory_manager.h>
#include <starpu_stdlib.h>

static size_t global_size[STARPU_MAXNODES];
static size_t used_size[STARPU_MAXNODES];

/* This is used as an optimization to avoid to wake up allocating threads for
 * each and every deallocation, only to find that there is still not enough
 * room.  */
/* Minimum amount being waited for */
static size_t min_waiting_size[STARPU_MAXNODES];
/* Number of waiters */
static int waiters[STARPU_MAXNODES];

static starpu_pthread_mutex_t lock_nodes[STARPU_MAXNODES];
static starpu_pthread_cond_t cond_nodes[STARPU_MAXNODES];

int _starpu_memory_manager_init()
{
	int i;

	for(i=0 ; i<STARPU_MAXNODES ; i++)
	{
		global_size[i] = 0;
		used_size[i] = 0;
		min_waiting_size[i] = 0;
		STARPU_PTHREAD_MUTEX_INIT(&lock_nodes[i], NULL);
		STARPU_PTHREAD_COND_INIT(&cond_nodes[i], NULL);
	}
	return 0;
}

void _starpu_memory_manager_set_global_memory_size(unsigned node, size_t size)
{
	STARPU_PTHREAD_MUTEX_LOCK(&lock_nodes[node]);
	if (!global_size[node]) {
		global_size[node] = size;
		_STARPU_DEBUG("Global size for node %d is %ld\n", node, (long)global_size[node]);
	} else {
		STARPU_ASSERT(global_size[node] == size);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&lock_nodes[node]);
}

size_t _starpu_memory_manager_get_global_memory_size(unsigned node)
{
	return global_size[node];
}


int _starpu_memory_manager_can_allocate_size(size_t size, unsigned node)
{
	int ret;

	STARPU_PTHREAD_MUTEX_LOCK(&lock_nodes[node]);
	if (global_size[node] == 0)
	{
		// We do not have information on the available size, let's suppose it is going to fit
		used_size[node] += size;
		ret = 1;
	}
	else if (used_size[node] + size <= global_size[node])
	{
		used_size[node] += size;
		ret = 1;
	}
	else
	{
		ret = 0;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&lock_nodes[node]);
	return ret;
}

void _starpu_memory_manager_deallocate_size(size_t size, unsigned node)
{
	STARPU_PTHREAD_MUTEX_LOCK(&lock_nodes[node]);

	used_size[node] -= size;

	/* If there's now room for waiters, wake them */
	if (min_waiting_size[node] &&
		global_size[node] - used_size[node] >= min_waiting_size[node])
		STARPU_PTHREAD_COND_BROADCAST(&cond_nodes[node]);

	STARPU_PTHREAD_MUTEX_UNLOCK(&lock_nodes[node]);
}

starpu_ssize_t starpu_memory_get_total(unsigned node)
{
	if (global_size[node] == 0)
		return -1;
	else
		return global_size[node];
}

starpu_ssize_t starpu_memory_get_available(unsigned node)
{
	if (global_size[node] == 0)
		return -1;
	else
		return global_size[node] - used_size[node];
}

void starpu_memory_wait_available(unsigned node, size_t size)
{
	STARPU_PTHREAD_MUTEX_LOCK(&lock_nodes[node]);
	waiters[node]++;

	/* Tell deallocators we need this amount */
	if (!min_waiting_size[node] || size < min_waiting_size[node])
		min_waiting_size[node] = size;

	/* Wait for it */
	while (used_size[node] + size > global_size[node])
		STARPU_PTHREAD_COND_WAIT(&cond_nodes[node], &lock_nodes[node]);

	if (!--waiters[node])
		/* Nobody is waiting any more, we can reset the minimum
		 */
		min_waiting_size[node] = 0;
	STARPU_PTHREAD_MUTEX_UNLOCK(&lock_nodes[node]);
}
