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

#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <common/list.h>

/** @file */

MULTILIST_CREATE_TYPE(_starpu_graph_node, all)
MULTILIST_CREATE_TYPE(_starpu_graph_node, top)
MULTILIST_CREATE_TYPE(_starpu_graph_node, bottom)
MULTILIST_CREATE_TYPE(_starpu_graph_node, dropped)

struct _starpu_graph_node
{
	/** protects access to the job */
	starpu_pthread_mutex_t mutex;
	/** pointer to the job, if it is still alive, NULL otherwise */
	struct _starpu_job *job;

	/**
	 * Fields for graph analysis for scheduling heuristics
	 */
	/** Member of list of all jobs without incoming dependency */
	struct _starpu_graph_node_multilist_top top;
	/** Member of list of all jobs without outgoing dependency */
	struct _starpu_graph_node_multilist_bottom bottom;
	/** Member of list of all jobs */
	struct _starpu_graph_node_multilist_all all;
	/** Member of list of dropped jobs */
	struct _starpu_graph_node_multilist_dropped dropped;

	/** set of incoming dependencies */
	/** May contain NULLs for terminated jobs */
	struct _starpu_graph_node **incoming;
	/** Index within corresponding outgoing array */
	unsigned *incoming_slot;
	/** Number of slots used */
	unsigned n_incoming;
	/** Size of incoming */
	unsigned alloc_incoming;
	/** set of outgoing dependencies */
	struct _starpu_graph_node **outgoing;

	/** Index within corresponding incoming array */
	unsigned *outgoing_slot;
	/** Number of slots used */
	unsigned n_outgoing;
	/** Size of outgoing */
	unsigned alloc_outgoing;

	/** Rank from bottom, in number of jobs
	 * Only available if _starpu_graph_compute_depths was called
	 */
	unsigned depth;
	/** Number of children, grand-children, etc.
	 * Only available if _starpu_graph_compute_descendants was called
	 */
	unsigned descendants;

	/** Variable available for graph flow */
	int graph_n;
};

MULTILIST_CREATE_INLINES(struct _starpu_graph_node, _starpu_graph_node, all)
MULTILIST_CREATE_INLINES(struct _starpu_graph_node, _starpu_graph_node, top)
MULTILIST_CREATE_INLINES(struct _starpu_graph_node, _starpu_graph_node, bottom)
MULTILIST_CREATE_INLINES(struct _starpu_graph_node, _starpu_graph_node, dropped)

extern int _starpu_graph_record;
void _starpu_graph_init(void);
void _starpu_graph_wrlock(void);
void _starpu_graph_rdlock(void);
void _starpu_graph_wrunlock(void);
void _starpu_graph_rdunlock(void);

/** Add a job to the graph, called before any _starpu_graph_add_job_dep call */
void _starpu_graph_add_job(struct _starpu_job *job);

/** Add a dependency between jobs */
void _starpu_graph_add_job_dep(struct _starpu_job *job, struct _starpu_job *prev_job);

/** Remove a job from the graph */
void _starpu_graph_drop_job(struct _starpu_job *job);

/** Really drop the nodes from the graph now */
void _starpu_graph_drop_dropped_nodes(void);

/**
 * This make StarPU compute for each task the depth, i.e. the length
 * of the longest path to a task without outgoing dependencies.
 * This does not take job duration into account, just the number
*/
void _starpu_graph_compute_depths(void);

/** Compute the descendants of jobs in the graph */
void _starpu_graph_compute_descendants(void);

/**
 * This calls \e func for each node of the task graph, passing also \e
 * data as it
 * Apply func on each job of the graph
*/
void _starpu_graph_foreach(void (*func)(void *data, struct _starpu_graph_node *node), void *data);

#endif /* __GRAPH_H__ */
