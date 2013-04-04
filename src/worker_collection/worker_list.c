/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013  Université de Bordeaux 1
 * Copyright (C) 2012-2013  Centre National de la Recherche Scientifique
 * Copyright (C) 2011-2013  INRIA
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

static unsigned list_has_next(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it)
{
	int nworkers = (int)workers->nworkers;
	STARPU_ASSERT(it != NULL);

	unsigned ret = it->cursor < nworkers ;

	if(!ret) it->cursor = 0;

	return ret;
}

static int list_get_next(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it)
{
	int *workerids = (int *)workers->workerids;
	int nworkers = (int)workers->nworkers;

	STARPU_ASSERT(it->cursor < nworkers);

	int ret = workerids[(it->cursor)++];

	return ret;
}

static unsigned _worker_belongs_to_ctx(struct starpu_worker_collection *workers, int workerid)
{
	int *workerids = (int *)workers->workerids;
	unsigned nworkers = workers->nworkers;
	
	unsigned i;
	for(i = 0; i < nworkers; i++)
	{
		if(workerids[i] == workerid)
			return 1;
	}
	return 0;
}

static int list_add(struct starpu_worker_collection *workers, int worker)
{
	int *workerids = (int *)workers->workerids;
	unsigned *nworkers = &workers->nworkers;

	STARPU_ASSERT(*nworkers < STARPU_NMAXWORKERS - 1);

	if(!_worker_belongs_to_ctx(workers, worker))
	{
		workerids[(*nworkers)++] = worker;
		return worker;
	}
	else 
		return -1;
}

static int _get_first_free_worker(int *workerids, int nworkers)
{
	int i;
	for(i = 0; i < nworkers; i++)
		if(workerids[i] == -1)
			return i;

	return -1;
}

/* rearange array of workerids in order not to have {-1, -1, 5, -1, 7}
   and have instead {5, 7, -1, -1, -1} 
   it is easier afterwards to iterate the array
*/
static void _rearange_workerids(int *workerids, int old_nworkers)
{
	int first_free_id = -1;
	int i;
	for(i = 0; i < old_nworkers; i++)
	{
		if(workerids[i] != -1)
		{
			first_free_id = _get_first_free_worker(workerids, old_nworkers);
			if(first_free_id != -1)
			{
				workerids[first_free_id] = workerids[i];
				workerids[i] = -1;
			}
		}
	  }
}

static int list_remove(struct starpu_worker_collection *workers, int worker)
{
	int *workerids = (int *)workers->workerids;
	unsigned nworkers = workers->nworkers;
	
	int found_worker = -1;
	unsigned i;
	for(i = 0; i < nworkers; i++)
	{
		if(workerids[i] == worker)
		{
			workerids[i] = -1;
			found_worker = worker;
			break;
		}
	}

	_rearange_workerids(workerids, nworkers);
	if(found_worker != -1)
		workers->nworkers--;

	return found_worker;
}

static void _init_workers(int *workerids)
{
	unsigned i;
	for(i = 0; i < STARPU_NMAXWORKERS; i++)
		workerids[i] = -1;
	return;
}

static void list_init(struct starpu_worker_collection *workers)
{
	int *workerids = (int*)malloc(STARPU_NMAXWORKERS * sizeof(int));
	_init_workers(workerids);

	workers->workerids = (void*)workerids;
	workers->nworkers = 0;

	return;
}

static void list_deinit(struct starpu_worker_collection *workers)
{
	free(workers->workerids);
}

static void list_init_iterator(struct starpu_worker_collection *workers STARPU_ATTRIBUTE_UNUSED, struct starpu_sched_ctx_iterator *it)
{
	*((int*)it) = 0;
}

struct starpu_worker_collection worker_list =
{
	.has_next = list_has_next,
	.get_next = list_get_next,
	.add = list_add,
	.remove = list_remove,
	.init = list_init,
	.deinit = list_deinit,
	.init_iterator = list_init_iterator,
	.type = STARPU_WORKER_LIST
};

