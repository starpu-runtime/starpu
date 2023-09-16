/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __MEMORY_NODES_H__
#define __MEMORY_NODES_H__

/** @file */

#include <starpu.h>
#include <common/config.h>
#include <datawizard/coherency.h>
#include <datawizard/memalloc.h>
#include <datawizard/node_ops.h>
#include <common/utils.h>
#include <core/workers.h>

#ifdef STARPU_SIMGRID
#include <core/simgrid.h>
#endif

#pragma GCC visibility push(hidden)

extern char _starpu_worker_drives_memory[STARPU_NMAXWORKERS][STARPU_MAXNODES];

struct _starpu_cond_and_worker
{
	starpu_pthread_cond_t *cond;
	struct _starpu_worker *worker;
};

// TODO: split out all these arrays into struct _starpu_node
struct _starpu_memory_node_descr
{
	unsigned nnodes;
	enum starpu_node_kind nodes[STARPU_MAXNODES];
	const struct _starpu_node_ops *node_ops[STARPU_MAXNODES];

	/** Get the device id associated to this node, or -1 if not applicable */
	int devid[STARPU_MAXNODES];

	unsigned nworkers[STARPU_MAXNODES];

#ifdef STARPU_SIMGRID
	starpu_sg_host_t host[STARPU_MAXNODES];
#endif

	// TODO move this 2 lists outside struct _starpu_memory_node_descr
	/** Every worker is associated to a condition variable on which the
	 * worker waits when there is task available. It is possible that
	 * multiple worker share the same condition variable, so we maintain a
	 * list of all these condition variables so that we can wake up all
	 * worker attached to a memory node that are waiting on a task. */
	starpu_pthread_rwlock_t conditions_rwlock;
	struct _starpu_cond_and_worker conditions_attached_to_node[STARPU_MAXNODES][STARPU_NMAXWORKERS];
	struct _starpu_cond_and_worker conditions_all[STARPU_MAXNODES*STARPU_NMAXWORKERS];
	/** the number of queues attached to each node */
	unsigned total_condition_count;
	unsigned condition_count[STARPU_MAXNODES];
	unsigned mapped[STARPU_MAXNODES];
};

extern struct _starpu_memory_node_descr _starpu_descr;

void _starpu_memory_nodes_init(void);
void _starpu_memory_nodes_deinit(void);

/** Record that there is an additional worker that uses this memory node */
static inline void _starpu_memory_node_add_nworkers(unsigned node)
{
	_starpu_descr.nworkers[node]++;
}

/** Record that this worker will driver data transfers for this memory node. */
void _starpu_worker_drives_memory_node(struct _starpu_worker *worker, unsigned memnode);

static inline const struct _starpu_node_ops *_starpu_memory_node_get_node_ops(unsigned node)
{
	return _starpu_descr.node_ops[node];
}

/** Get the number of workers that use this memory node */
static inline unsigned _starpu_memory_node_get_nworkers(unsigned node)
{
	return _starpu_descr.nworkers[node];
}

#ifdef STARPU_SIMGRID
static inline void _starpu_simgrid_memory_node_set_host(unsigned node, starpu_sg_host_t host)
{
	_starpu_descr.host[node] = host;
}

static inline starpu_sg_host_t _starpu_simgrid_memory_node_get_host(unsigned node)
{
	return _starpu_descr.host[node];
}
#endif

/** Note that this memory node can map CPU data */
void _starpu_memory_node_set_mapped(unsigned node);
/** Returns whether this memory node can map CPU data */
unsigned _starpu_memory_node_get_mapped(unsigned node);

/** Registers a memory node. Returns the memory node number */
unsigned _starpu_memory_node_register(enum starpu_node_kind kind, int devid);

//void _starpu_memory_node_attach_queue(struct starpu_jobq_s *q, unsigned nodeid);
/** Register a condition variable associated to worker which is associated to a
 * memory node itself. */
void _starpu_memory_node_register_condition(struct _starpu_worker *worker, starpu_pthread_cond_t *cond, unsigned nodeid);

/** See starpu_memory_node_get_description() */
static inline struct _starpu_memory_node_descr *_starpu_memory_node_get_description(void)
{
	return &_starpu_descr;
}

#define _starpu_node_needs_map_update(node) \
	(starpu_node_get_kind(node) == STARPU_OPENCL_RAM)

/** See starpu_node_get_kind() */
static inline enum starpu_node_kind _starpu_node_get_kind(unsigned node)
{
	return _starpu_descr.nodes[node];
}
#define starpu_node_get_kind _starpu_node_get_kind

#if STARPU_MAXNODES == 1
#define _starpu_memory_nodes_get_count() 1
#else
/** See starpu_memory_nodes_get_count() */
static inline unsigned _starpu_memory_nodes_get_count(void)
{
	return _starpu_descr.nnodes;
}
#endif
#define starpu_memory_nodes_get_count _starpu_memory_nodes_get_count

#if STARPU_MAXNODES == 1
#define _starpu_worker_get_memory_node(workerid) 0
#else
/** See starpu_worker_get_memory_node() */
static inline unsigned _starpu_worker_get_memory_node(unsigned workerid)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();

	/** This workerid may either be a basic worker or a combined worker */
	unsigned nworkers = config->topology.nworkers;

	if (workerid < config->topology.nworkers)
		return config->workers[workerid].memory_node;

	/** We have a combined worker */
	unsigned ncombinedworkers STARPU_ATTRIBUTE_UNUSED = config->topology.ncombinedworkers;
	STARPU_ASSERT_MSG(workerid < ncombinedworkers + nworkers, "Bad workerid %u, maximum %u", workerid, ncombinedworkers + nworkers);
	return config->combined_workers[workerid - nworkers].memory_node;

}
#endif
#define starpu_worker_get_memory_node _starpu_worker_get_memory_node

#if STARPU_MAXNODES == 1
#define _starpu_worker_get_local_memory_node() 0
#else
/** See starpu_worker_get_local_memory_node */
static inline unsigned _starpu_worker_get_local_memory_node(void)
{
	struct _starpu_worker *worker = _starpu_get_local_worker_key();
	if (!worker)
		return STARPU_MAIN_RAM;
	return worker->memory_node;
}
#endif
#define starpu_worker_get_local_memory_node _starpu_worker_get_local_memory_node

#pragma GCC visibility pop

#endif // __MEMORY_NODES_H__
