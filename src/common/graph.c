/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * This stores the task graph structure, to used by the schedulers which need
 * it.  We do not always enable it since it is costly.  To avoid interfering
 * too much with execution, it may be a bit outdated, i.e. still contain jobs
 * which have completed very recently.
 *
 * This is because we drop nodes lazily: when a job terminates, we just add the
 * node to the dropped list (to avoid having to take the mutex on the whole
 * graph).  The graph gets updated whenever the graph mutex becomes available.
 */

#include <starpu.h>
#include <core/jobs.h>
#include <common/graph.h>
#include <core/workers.h>

/* Protects the whole task graph except the dropped list */
static starpu_pthread_rwlock_t graph_lock;

/* Whether we should enable recording the task graph */
int _starpu_graph_record;

/* This list contains all nodes without incoming dependency */
struct _starpu_graph_node_multilist_top top;
/* This list contains all nodes without outgoing dependency */
struct _starpu_graph_node_multilist_bottom bottom;
/* This list contains all nodes */
struct _starpu_graph_node_multilist_all all;

/* Protects the dropped list, always taken before graph lock */
static starpu_pthread_mutex_t dropped_lock;
/* This list contains all dropped nodes, i.e. the job terminated by the corresponding node is still int he graph */
struct _starpu_graph_node_multilist_dropped dropped;

void _starpu_graph_init(void)
{
	STARPU_PTHREAD_RWLOCK_INIT(&graph_lock, NULL);
	_starpu_graph_node_multilist_head_init_top(&top);
	_starpu_graph_node_multilist_head_init_bottom(&bottom);
	_starpu_graph_node_multilist_head_init_all(&all);
	STARPU_PTHREAD_MUTEX_INIT(&dropped_lock, NULL);
	_starpu_graph_node_multilist_head_init_dropped(&dropped);
}

/* LockWR the graph lock */
void _starpu_graph_wrlock(void)
{
	starpu_worker_relax_on();
	STARPU_PTHREAD_RWLOCK_WRLOCK(&graph_lock);
	starpu_worker_relax_off();
}

void _starpu_graph_drop_node(struct _starpu_graph_node *node);

/* This flushes the list of nodes to be dropped. Both the dropped_lock and
 * graph_lock mutexes have to be held on entry, and are released.  */
void _starpu_graph_drop_dropped_nodes(void)
{
	struct _starpu_graph_node_multilist_dropped dropping;

	/* Pick up the list of dropped nodes */
	_starpu_graph_node_multilist_move_dropped(&dropped, &dropping);
	STARPU_PTHREAD_MUTEX_UNLOCK(&dropped_lock);

	/* And now process it if it's not empty.  */
	if (!_starpu_graph_node_multilist_empty_dropped(&dropping))
	{
		struct _starpu_graph_node *node, *next;

		for (node = _starpu_graph_node_multilist_begin_dropped(&dropping);
		     node != _starpu_graph_node_multilist_end_dropped(&dropping);
		     node = next)
		{
			next = _starpu_graph_node_multilist_next_dropped(node);
			_starpu_graph_drop_node(node);
		}
	}
	STARPU_PTHREAD_RWLOCK_UNLOCK(&graph_lock);
}

/* UnlockWR the graph lock */
void _starpu_graph_wrunlock(void)
{
	starpu_worker_relax_on();
	STARPU_PTHREAD_MUTEX_LOCK(&dropped_lock);
	starpu_worker_relax_off();
	_starpu_graph_drop_dropped_nodes();
}

/* LockRD the graph lock */
void _starpu_graph_rdlock(void)
{
	starpu_worker_relax_on();
	STARPU_PTHREAD_RWLOCK_RDLOCK(&graph_lock);
	starpu_worker_relax_off();
}

/* UnlockRD the graph lock */
void _starpu_graph_rdunlock(void)
{
	STARPU_PTHREAD_RWLOCK_UNLOCK(&graph_lock);
	/* Take the opportunity to try to take it WR */
	if (STARPU_PTHREAD_RWLOCK_TRYWRLOCK(&graph_lock) == 0)
		/* Good, flush dropped nodes */
		_starpu_graph_wrunlock();
}

static void __starpu_graph_foreach(void (*func)(void *data, struct _starpu_graph_node *node), void *data)
{
	struct _starpu_graph_node *node;

	for (node = _starpu_graph_node_multilist_begin_all(&all);
	     node != _starpu_graph_node_multilist_end_all(&all);
	     node = _starpu_graph_node_multilist_next_all(node))
		func(data, node);
}

/* Add a node to the graph */
void _starpu_graph_add_job(struct _starpu_job *job)
{
	struct _starpu_graph_node *node;
	_STARPU_CALLOC(node, 1, sizeof(*node));
	node->job = job;
	job->graph_node = node;
	STARPU_PTHREAD_MUTEX_INIT(&node->mutex, NULL);

	_starpu_graph_wrlock();

	/* It does not have any dependency yet, add to all lists */
	_starpu_graph_node_multilist_push_back_top(&top, node);
	_starpu_graph_node_multilist_push_back_bottom(&bottom, node);
	_starpu_graph_node_multilist_push_back_all(&all, node);

	_starpu_graph_wrunlock();
}

/* Add a node to an array of nodes */
static unsigned add_node(struct _starpu_graph_node *node, struct _starpu_graph_node ***nodes, unsigned *n_nodes, unsigned *alloc_nodes, unsigned **slot)
{
	unsigned ret;
	if (*n_nodes == *alloc_nodes)
	{
		if (*alloc_nodes)
			*alloc_nodes *= 2;
		else
			*alloc_nodes = 4;
		_STARPU_REALLOC(*nodes, *alloc_nodes * sizeof(**nodes));
		if (slot)
		{
			_STARPU_REALLOC(*slot, *alloc_nodes * sizeof(**slot));
		}
	}
	ret = (*n_nodes)++;
	(*nodes)[ret] = node;
	return ret;
}

/* Add a dependency between nodes */
void _starpu_graph_add_job_dep(struct _starpu_job *job, struct _starpu_job *prev_job)
{
	unsigned rank_incoming, rank_outgoing;
	_starpu_graph_wrlock();
	struct _starpu_graph_node *node = job->graph_node;
	struct _starpu_graph_node *prev_node = prev_job->graph_node;
	if (!node || !prev_node)
	{
		/* Already gone */
		_starpu_graph_wrunlock();
		return;
	}

	if (_starpu_graph_node_multilist_queued_bottom(prev_node))
		/* Previous node is not at bottom any more */
		_starpu_graph_node_multilist_erase_bottom(&bottom, prev_node);

	if (_starpu_graph_node_multilist_queued_top(node))
		/* Next node is not at top any more */
		_starpu_graph_node_multilist_erase_top(&top, node);

	rank_incoming = add_node(prev_node, &node->incoming, &node->n_incoming, &node->alloc_incoming, &node->incoming_slot);
	rank_outgoing = add_node(node, &prev_node->outgoing, &prev_node->n_outgoing, &prev_node->alloc_outgoing, &prev_node->outgoing_slot);
	prev_node->outgoing_slot[rank_outgoing] = rank_incoming;
	node->incoming_slot[rank_incoming] = rank_outgoing;

	_starpu_graph_wrunlock();
}

/* Drop a node, and thus its dependencies */
void _starpu_graph_drop_node(struct _starpu_graph_node *node)
{
	unsigned i;
	STARPU_ASSERT(!node->job);

	if (_starpu_graph_node_multilist_queued_bottom(node))
		_starpu_graph_node_multilist_erase_bottom(&bottom, node);
	if (_starpu_graph_node_multilist_queued_top(node))
		_starpu_graph_node_multilist_erase_top(&top, node);
	if (_starpu_graph_node_multilist_queued_all(node))
		_starpu_graph_node_multilist_erase_all(&all, node);

	/* Drop ourself from the incoming part of the outgoing nodes.  */
	for (i = 0; i < node->n_outgoing; i++)
	{
		struct _starpu_graph_node *next = node->outgoing[i];
		if (next)
			next->incoming[node->outgoing_slot[i]] = NULL;
	}

	/* Drop ourself from the outgoing part of the incoming nodes,
	 * in case we happen to get dropped before it.  */
	for (i = 0; i < node->n_incoming; i++)
	{
		struct _starpu_graph_node *prev = node->incoming[i];
		if (prev)
			prev->outgoing[node->incoming_slot[i]] = NULL;
	}

	node->n_outgoing = 0;
	free(node->outgoing);
	node->outgoing = NULL;
	free(node->outgoing_slot);
	node->outgoing_slot = NULL;
	node->alloc_outgoing = 0;
	node->n_incoming = 0;
	free(node->incoming);
	node->incoming = NULL;
	free(node->incoming_slot);
	node->incoming_slot = NULL;
	node->alloc_incoming = 0;
	free(node);
}

/* Drop a job */
void _starpu_graph_drop_job(struct _starpu_job *job)
{
	struct _starpu_graph_node *node = job->graph_node;
	job->graph_node = NULL;
	if (!node)
		return;

	starpu_worker_relax_on();
	STARPU_PTHREAD_MUTEX_LOCK(&node->mutex);
	starpu_worker_relax_off();
	/* Will not be able to use the job any more */
	node->job = NULL;
	STARPU_PTHREAD_MUTEX_UNLOCK(&node->mutex);

	starpu_worker_relax_on();
	STARPU_PTHREAD_MUTEX_LOCK(&dropped_lock);
	starpu_worker_relax_off();
	/* Queue for removal when lock becomes available */
	_starpu_graph_node_multilist_push_back_dropped(&dropped, node);
	if (STARPU_PTHREAD_RWLOCK_TRYWRLOCK(&graph_lock) == 0)
	{
		/* Graph wrlock is available, drop nodes immediately */
		_starpu_graph_drop_dropped_nodes();
	}
	else
		STARPU_PTHREAD_MUTEX_UNLOCK(&dropped_lock);
}

static void _starpu_graph_set_n(void *data, struct _starpu_graph_node *node)
{
	int value = (intptr_t) data;
	node->graph_n = value;
}

/* Call func for each vertex of the task graph, from bottom to top, in topological order */
static void _starpu_graph_compute_bottom_up(void (*func)(struct _starpu_graph_node *next_node, struct _starpu_graph_node *prev_node, void *data), void *data)
{
	struct _starpu_graph_node *node, *node2;
	struct _starpu_graph_node **current_set = NULL, **next_set = NULL, **swap_set;
	unsigned current_n, next_n, i, j;
	unsigned current_alloc = 0, next_alloc = 0, swap_alloc;

	/* Classical flow algorithm: start from bottom, and propagate depths to top */

	/* Set number of processed outgoing edges to 0 for each node */
	__starpu_graph_foreach(_starpu_graph_set_n, (void*) 0);

	/* Start with the bottom of the graph */
	current_n = 0;
	for (node = _starpu_graph_node_multilist_begin_bottom(&bottom);
	     node != _starpu_graph_node_multilist_end_bottom(&bottom);
	     node = _starpu_graph_node_multilist_next_bottom(node))
		add_node(node, &current_set, &current_n, &current_alloc, NULL);

	/* Now propagate to top as long as we have current nodes */
	while (current_n)
	{
		/* Next set is initially empty */
		next_n = 0;

		/* For each node in the current set */
		for (i = 0; i < current_n; i++)
		{
			node = current_set[i];
			/* For each parent of this node */
			for (j = 0; j < node->n_incoming; j++)
			{
				node2 = node->incoming[j];
				if (!node2)
					continue;
				node2->graph_n++;
				func(node, node2, data);

				if ((unsigned) node2->graph_n == node2->n_outgoing)
					/* All outgoing edges were processed, can now add to next set */
					add_node(node2, &next_set, &next_n, &next_alloc, NULL);
			}
		}

		/* Swap next set with current set */
		swap_set = next_set;
		swap_alloc = next_alloc;
		next_set = current_set;
		next_alloc = current_alloc;
		current_set = swap_set;
		current_alloc = swap_alloc;
		current_n = next_n;
	}
	free(current_set);
	free(next_set);
}

static void compute_depth(struct _starpu_graph_node *next_node, struct _starpu_graph_node *prev_node, void *data)
{
	(void)data;
	if (prev_node->depth < next_node->depth + 1)
		prev_node->depth = next_node->depth + 1;
}

void _starpu_graph_compute_depths(void)
{
	struct _starpu_graph_node *node;

	_starpu_graph_wrlock();

	/* The bottom of the graph has depth 0 */
	for (node = _starpu_graph_node_multilist_begin_bottom(&bottom);
	     node != _starpu_graph_node_multilist_end_bottom(&bottom);
	     node = _starpu_graph_node_multilist_next_bottom(node))
		node->depth = 0;

	_starpu_graph_compute_bottom_up(compute_depth, NULL);

	_starpu_graph_wrunlock();
}

void _starpu_graph_compute_descendants(void)
{
	struct _starpu_graph_node *node, *node2, *node3;
	struct _starpu_graph_node **current_set = NULL, **next_set = NULL, **swap_set;
	unsigned current_n, next_n, i, j;
	unsigned current_alloc = 0, next_alloc = 0, swap_alloc;

	_starpu_graph_wrlock();

	/* Yes, this is O(|V|.(|V|+|E|)) :( */

	/* We could get O(|V|.|E|) by doing a topological sort first.
	 *
	 * |E| is usually O(|V|), though (bounded number of data dependencies,
	 * and we use synchronization tasks) */

	for (node = _starpu_graph_node_multilist_begin_all(&all);
	     node != _starpu_graph_node_multilist_end_all(&all);
	     node = _starpu_graph_node_multilist_next_all(node))
	{
		unsigned descendants;

		/* Mark all nodes as unseen */
		for (node2 = _starpu_graph_node_multilist_begin_all(&all);
		     node2 != _starpu_graph_node_multilist_end_all(&all);
		     node2 = _starpu_graph_node_multilist_next_all(node2))
			node2->graph_n = 0;

		/* Start with the node we want to compute the number of descendants of */
		current_n = 0;
		add_node(node, &current_set, &current_n, &current_alloc, NULL);
		node->graph_n = 1;

		descendants = 0;
		/* While we have descendants, count their descendants */
		while (current_n)
		{
			/* Next set is initially empty */
			next_n = 0;

			/* For each node in the current set */
			for (i = 0; i < current_n; i++)
			{
				node2 = current_set[i];
				/* For each child of this node2 */
				for (j = 0; j < node2->n_outgoing; j++)
				{
					node3 = node2->outgoing[j];
					if (!node3)
						continue;
					if (node3->graph_n)
						/* Already seen */
						continue;
					/* Add this node */
					node3->graph_n = 1;
					descendants++;
					add_node(node3, &next_set, &next_n, &next_alloc, NULL);
				}
			}
			/* Swap next set with current set */
			swap_set = next_set;
			swap_alloc = next_alloc;
			next_set = current_set;
			next_alloc = current_alloc;
			current_set = swap_set;
			current_alloc = swap_alloc;
			current_n = next_n;
		}
		node->descendants = descendants;
	}

	_starpu_graph_wrunlock();

	free(current_set);
	free(next_set);
}

void _starpu_graph_foreach(void (*func)(void *data, struct _starpu_graph_node *node), void *data)
{
	_starpu_graph_wrlock();
	__starpu_graph_foreach(func, data);
	_starpu_graph_wrunlock();
}
