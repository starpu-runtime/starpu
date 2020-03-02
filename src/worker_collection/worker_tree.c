/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#include "core/workers.h"

static unsigned tree_has_next_unblocked_worker(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it)
{
	STARPU_ASSERT(it != NULL);
	if(workers->nworkers == 0)
		return 0;

	struct starpu_tree *tree = (struct starpu_tree*)workers->collection_private;
	struct starpu_tree *neighbour = starpu_tree_get_neighbour(tree, (struct starpu_tree*)it->value, it->visited, workers->present);

	if(!neighbour)
	{
		starpu_tree_reset_visited(tree, it->visited);
		it->value = NULL;
		it->possible_value = NULL;
		return 0;
	}
	int id = -1;
	int *workerids;
	int nworkers = starpu_bindid_get_workerids(neighbour->id, &workerids);
	int w;
	for(w = 0; w < nworkers; w++)
	{
		if(!it->visited[workerids[w]] && workers->present[workerids[w]])
		{
			if(workers->is_unblocked[workerids[w]])
			{
				id = workerids[w];
				it->possible_value = neighbour;
				break;
			}
			else
			{
				it->visited[workerids[w]] = 1;
				it->value = neighbour;

				return tree_has_next_unblocked_worker(workers, it);
			}
		}
	}

	STARPU_ASSERT_MSG(id != -1, "bind id (%d) for workerid (%d) not correct", neighbour->id, id);

	return 1;
}

static int tree_get_next_unblocked_worker(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it)
{
	int ret = -1;

	struct starpu_tree *tree = (struct starpu_tree *)workers->collection_private;
	struct starpu_tree *neighbour = NULL;
	if(it->possible_value)
	{
		neighbour = it->possible_value;
		it->possible_value = NULL;
	}
	else
		neighbour = starpu_tree_get_neighbour(tree, (struct starpu_tree*)it->value, it->visited, workers->present);

	STARPU_ASSERT_MSG(neighbour, "no element anymore");


	int *workerids;
	int nworkers = starpu_bindid_get_workerids(neighbour->id, &workerids);
	int w;
	for(w = 0; w < nworkers; w++)
	{
		if(!it->visited[workerids[w]] && workers->present[workerids[w]] && workers->is_unblocked[workerids[w]])
		{
			ret = workerids[w];
			it->visited[workerids[w]] = 1;
			it->value = neighbour;
			break;
		}
	}
	STARPU_ASSERT_MSG(ret != -1, "bind id not correct");
	return ret;
}

static unsigned tree_has_next_master(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it)
{
	STARPU_ASSERT(it != NULL);
	if(workers->nworkers == 0)
		return 0;

	struct starpu_tree *tree = (struct starpu_tree*)workers->collection_private;
	struct starpu_tree *neighbour = starpu_tree_get_neighbour(tree, (struct starpu_tree*)it->value, it->visited, workers->is_master);

	if(!neighbour)
	{
		starpu_tree_reset_visited(tree, it->visited);
		it->value = NULL;
		it->possible_value = NULL;
		return 0;
	}
	int id = -1;
	int *workerids;
	int nworkers = starpu_bindid_get_workerids(neighbour->id, &workerids);
	int w;
	for(w = 0; w < nworkers; w++)
	{
		if(!it->visited[workerids[w]] && workers->is_master[workerids[w]])
		{
			id = workerids[w];
			it->possible_value = neighbour;
			break;
		}
	}

	STARPU_ASSERT_MSG(id != -1, "bind id (%d) for workerid (%d) not correct", neighbour->id, id);

	return 1;
}

static int tree_get_next_master(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it)
{
	int ret = -1;

	struct starpu_tree *tree = (struct starpu_tree *)workers->collection_private;
	struct starpu_tree *neighbour = NULL;
	if(it->possible_value)
	{
		neighbour = it->possible_value;
		it->possible_value = NULL;
	}
	else
		neighbour = starpu_tree_get_neighbour(tree, (struct starpu_tree*)it->value, it->visited, workers->is_master);

	STARPU_ASSERT_MSG(neighbour, "no element anymore");


	int *workerids;
	int nworkers = starpu_bindid_get_workerids(neighbour->id, &workerids);
	int w;
	for(w = 0; w < nworkers; w++)
	{
		if(!it->visited[workerids[w]] && workers->is_master[workerids[w]])
		{
			ret = workerids[w];
			it->visited[workerids[w]] = 1;
			it->value = neighbour;
			break;
		}
	}
	STARPU_ASSERT_MSG(ret != -1, "bind id not correct");

	return ret;
}

static unsigned tree_has_next(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it)
{
	if(it->possibly_parallel == 1)
                return tree_has_next_master(workers, it);
        else if(it->possibly_parallel == 0)
                return tree_has_next_unblocked_worker(workers, it);

	STARPU_ASSERT(it != NULL);
	if(workers->nworkers == 0)
		return 0;

	struct starpu_tree *tree = (struct starpu_tree*)workers->collection_private;
	int *workerids;
	int nworkers;
	int w;

	if (it->value)
	{
		struct starpu_tree *node = it->value;
		/* Are there workers left to be processed in the current node? */
		nworkers = starpu_bindid_get_workerids(node->id, &workerids);
		for(w = 0; w < nworkers; w++)
		{
			if(!it->visited[workerids[w]] && workers->present[workerids[w]] )
			{
				/* Still some! */
				it->possible_value = node;
				return 1;
			}
		}
	}

	struct starpu_tree *neighbour = starpu_tree_get_neighbour(tree, (struct starpu_tree*)it->value, it->visited, workers->present);

	if(!neighbour)
	{
		starpu_tree_reset_visited(tree, it->visited);
		it->value = NULL;
		it->possible_value = NULL;
		return 0;
	}
	int id = -1;
	nworkers = starpu_bindid_get_workerids(neighbour->id, &workerids);
	for(w = 0; w < nworkers; w++)
	{
		if(!it->visited[workerids[w]] && workers->present[workerids[w]])
		{
			id = workerids[w];
			it->possible_value = neighbour;
			break;
		}
	}

	STARPU_ASSERT_MSG(id != -1, "bind id (%d) for workerid (%d) not correct", neighbour->id, id);

	return 1;
}

static int tree_get_next(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it)
{
	if(it->possibly_parallel == 1)
                return tree_get_next_master(workers, it);
        else if(it->possibly_parallel == 0)
                return tree_get_next_unblocked_worker(workers, it);

	int ret = -1;

	struct starpu_tree *tree = (struct starpu_tree *)workers->collection_private;
	struct starpu_tree *neighbour = NULL;
	if(it->possible_value)
	{
		neighbour = it->possible_value;
		it->possible_value = NULL;
	}
	else
		neighbour = starpu_tree_get_neighbour(tree, (struct starpu_tree*)it->value, it->visited, workers->present);

	STARPU_ASSERT_MSG(neighbour, "no element anymore");


	int *workerids;
	int nworkers = starpu_bindid_get_workerids(neighbour->id, &workerids);
	int w;
	for(w = 0; w < nworkers; w++)
	{
		if(!it->visited[workerids[w]] && workers->present[workerids[w]] )
		{
			ret = workerids[w];
			it->visited[workerids[w]] = 1;
			it->value = neighbour;
			break;
		}
	}
	STARPU_ASSERT_MSG(ret != -1, "bind id not correct");

	return ret;
}

static int tree_add(struct starpu_worker_collection *workers, int worker)
{
	if(!workers->present[worker])
	{
		workers->present[worker] = 1;
		workers->workerids[workers->nworkers] = worker;
		workers->nworkers++;
		return worker;
	}
	else
		return -1;
}


static int tree_remove(struct starpu_worker_collection *workers, int worker)
{
	if(workers->present[worker])
	{
		unsigned i;
		for (i = 0; i < workers->nworkers; i++)
			if (workers->workerids[i] == worker)
			{
				memmove(&workers->workerids[i], &workers->workerids[i+1], (workers->nworkers-1-i) * sizeof(workers->workerids[i]));
				break;
			}
		workers->present[worker] = 0;
		workers->is_unblocked[worker] = 0;
		workers->is_master[worker] = 0;
		workers->nworkers--;
		return worker;
	}
	else
		return -1;
}

static void tree_init(struct starpu_worker_collection *workers)
{
	_STARPU_MALLOC(workers->workerids, (STARPU_NMAXWORKERS+STARPU_NMAX_COMBINEDWORKERS) * sizeof(int));
	workers->collection_private = (void*)starpu_workers_get_tree();
	workers->nworkers = 0;

	int i;
	int nworkers = starpu_worker_get_count();
	for(i = 0; i < nworkers; i++)
	{
		workers->workerids[i] = -1;
		workers->present[i] = 0;
		workers->is_unblocked[i] = 0;
		workers->is_master[i] = 0;
	}

	return;
}

static void tree_deinit(struct starpu_worker_collection *workers)
{
	(void) workers;
	free(workers->workerids);
}

static void tree_init_iterator(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it)
{
	(void) workers;
	it->value = NULL;
	it->possible_value = NULL;
	it->possibly_parallel = -1;
	int nworkers = starpu_worker_get_count();
	memset(&it->visited, 0, nworkers * sizeof(it->visited[0]));
}

static void tree_init_iterator_for_parallel_tasks(struct starpu_worker_collection *workers, struct starpu_sched_ctx_iterator *it, struct starpu_task *task)
{
	if (_starpu_get_nsched_ctxs() <= 1)
	{
		tree_init_iterator(workers, it);
		return;
	}
	tree_init_iterator(workers, it);
	it->possibly_parallel = task->possibly_parallel;
	int i;
	int nworkers = starpu_worker_get_count();
	for(i = 0; i < nworkers; i++)
	{
		workers->is_unblocked[i] = (workers->present[i] && !starpu_worker_is_blocked_in_parallel(i));
		if(!it->possibly_parallel) /* don't bother filling the table with masters we won't use it anyway */
			continue;
		workers->is_master[i] = (workers->present[i] && !starpu_worker_is_blocked_in_parallel(i) && !starpu_worker_is_slave_somewhere(i));
	}
}

struct starpu_worker_collection worker_tree =
{
	.has_next = tree_has_next,
	.get_next = tree_get_next,
	.add = tree_add,
	.remove = tree_remove,
	.init = tree_init,
	.deinit = tree_deinit,
	.init_iterator = tree_init_iterator,
	.init_iterator_for_parallel_tasks = tree_init_iterator_for_parallel_tasks,
	.type = STARPU_WORKER_TREE
};

#endif// STARPU_HAVE_HWLOC
