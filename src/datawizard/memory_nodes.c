/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <common/config.h>
#include <core/sched_policy.h>
#include <datawizard/datastats.h>
#include <datawizard/memory_manager.h>
#include <datawizard/memory_nodes.h>
#include <datawizard/malloc.h>
#include <common/fxt.h>
#include <datawizard/copy_driver.h>
#include <datawizard/memalloc.h>
#include <datawizard/node_ops.h>

char _starpu_worker_drives_memory[STARPU_NMAXWORKERS][STARPU_MAXNODES];

struct _starpu_memory_node_descr _starpu_descr;

void _starpu_memory_nodes_init(void)
{
	/* there is no node yet, subsequent nodes will be
	 * added using _starpu_memory_node_register */
	_starpu_descr.nnodes = 0;

	unsigned i;
	for (i = 0; i < STARPU_MAXNODES; i++)
	{
		_starpu_descr.nodes[i] = STARPU_UNUSED;
		_starpu_descr.nworkers[i] = 0;
	}
	memset(&_starpu_worker_drives_memory, 0, sizeof(_starpu_worker_drives_memory));
	STARPU_HG_DISABLE_CHECKING(_starpu_worker_drives_memory);

	_starpu_init_mem_chunk_lists();
	_starpu_init_data_request_lists();
	_starpu_memory_manager_init();

	STARPU_PTHREAD_RWLOCK_INIT(&_starpu_descr.conditions_rwlock, NULL);
	_starpu_descr.total_condition_count = 0;
}

void _starpu_memory_nodes_deinit(void)
{
	_starpu_deinit_data_request_lists();
	_starpu_deinit_mem_chunk_lists();

	STARPU_PTHREAD_RWLOCK_DESTROY(&_starpu_descr.conditions_rwlock);
}

#undef starpu_node_get_kind
enum starpu_node_kind starpu_node_get_kind(unsigned node)
{
	return _starpu_node_get_kind(node);
}

#undef starpu_memory_nodes_get_count
unsigned starpu_memory_nodes_get_count(void)
{
	return _starpu_memory_nodes_get_count();
}

unsigned starpu_memory_nodes_get_count_by_kind(enum starpu_node_kind kind)
{
	unsigned nnodes = _starpu_memory_nodes_get_count();
	unsigned id, cnt = 0;

	for (id = 0; id < nnodes; id++)
		if (_starpu_node_get_kind(id) == kind)
			cnt++;

	return cnt;
}

unsigned starpu_memory_node_get_ids_by_type(enum starpu_node_kind kind, unsigned *memory_nodes_ids, unsigned maxsize)
{
	unsigned nnodes = _starpu_memory_nodes_get_count();
	unsigned cnt = 0;
	unsigned id;

	for (id = 0; id < nnodes; id++)
	{
		if (_starpu_node_get_kind(id) == kind)
		{
			/* Perhaps the array is too small ? */
			if (cnt >= maxsize)
				return -ERANGE;

			memory_nodes_ids[cnt++] = id;
		}
	}

	return cnt;
}

int starpu_memory_node_get_name(unsigned node, char *name, size_t size)
{
	const char *prefix = _starpu_node_get_prefix(_starpu_descr.nodes[node]);
	return snprintf(name, size, "%s %d", prefix, _starpu_descr.devid[node]);
}

unsigned _starpu_memory_node_register(enum starpu_node_kind kind, int devid)
{
	const struct _starpu_node_ops *node_ops = starpu_memory_driver_info[kind].ops;
	unsigned node;
	/* ATOMIC_ADD returns the new value ... */
	node = STARPU_ATOMIC_ADD(&_starpu_descr.nnodes, 1) - 1;
	STARPU_ASSERT_MSG(node < STARPU_MAXNODES,"Too many nodes (%u) for maximum %d. Use configure option --enable-maxnodes=xxx to update the maximum number of nodes.", node + 1, STARPU_MAXNODES);

	_starpu_descr.nodes[node] = kind;
	_STARPU_TRACE_NEW_MEM_NODE(node);

	_starpu_descr.devid[node] = devid;
	_starpu_descr.node_ops[node] = node_ops;

	/* for now, there is no condition associated to that newly created node */
	_starpu_descr.condition_count[node] = 0;

	_starpu_malloc_init(node);

	return node;
}

/* TODO move in a more appropriate file  !! */
void _starpu_memory_node_register_condition(struct _starpu_worker *worker, starpu_pthread_cond_t *cond, unsigned nodeid)
{
	unsigned cond_id;
	unsigned nconds_total, nconds;

	STARPU_PTHREAD_RWLOCK_WRLOCK(&_starpu_descr.conditions_rwlock);

	/* we only insert the queue if it's not already in the list */
	nconds = _starpu_descr.condition_count[nodeid];
	for (cond_id = 0; cond_id < nconds; cond_id++)
	{
		if (_starpu_descr.conditions_attached_to_node[nodeid][cond_id].cond == cond)
		{
			STARPU_ASSERT(_starpu_descr.conditions_attached_to_node[nodeid][cond_id].worker == worker);

			/* the condition is already in the list */
			STARPU_PTHREAD_RWLOCK_UNLOCK(&_starpu_descr.conditions_rwlock);
			return;
		}
	}

	/* it was not found locally */
	_starpu_descr.conditions_attached_to_node[nodeid][cond_id].cond = cond;
	_starpu_descr.conditions_attached_to_node[nodeid][cond_id].worker = worker;
	_starpu_descr.condition_count[nodeid]++;

	/* do we have to add it in the global list as well ? */
	nconds_total = _starpu_descr.total_condition_count;
	for (cond_id = 0; cond_id < nconds_total; cond_id++)
	{
		if (_starpu_descr.conditions_all[cond_id].cond == cond)
		{
			/* the queue is already in the global list */
			STARPU_PTHREAD_RWLOCK_UNLOCK(&_starpu_descr.conditions_rwlock);
			return;
		}
	}

	/* it was not in the global list either */
	_starpu_descr.conditions_all[nconds_total].cond = cond;
	_starpu_descr.conditions_all[nconds_total].worker = worker;
	_starpu_descr.total_condition_count++;

	STARPU_PTHREAD_RWLOCK_UNLOCK(&_starpu_descr.conditions_rwlock);
}

void _starpu_memory_node_set_mapped(unsigned node)
{
	if (starpu_map_enabled() == 1)
		_starpu_descr.mapped[node] = 1;
#ifdef STARPU_VERBOSE
	else
		_STARPU_DISP("Warning: set_mapped requested on node %u, while map support is disabled\n", node);
#endif
}

unsigned _starpu_memory_node_get_mapped(unsigned node)
{
	return _starpu_descr.mapped[node];
}

#undef starpu_worker_get_memory_node
unsigned starpu_worker_get_memory_node(unsigned workerid)
{
	(void) workerid;
	return _starpu_worker_get_memory_node(workerid);
}

void _starpu_worker_drives_memory_node(struct _starpu_worker *worker, unsigned memnode)
{
	if (! _starpu_worker_drives_memory[worker->workerid][memnode])
	{
		_starpu_worker_drives_memory[worker->workerid][memnode] = 1;
#ifdef STARPU_SIMGRID
		starpu_pthread_queue_register(&worker->wait, &_starpu_simgrid_transfer_queue[memnode]);
#endif
		_starpu_memory_node_register_condition(worker, &worker->sched_cond, memnode);
	}
}

#undef starpu_worker_get_local_memory_node
unsigned starpu_worker_get_local_memory_node(void)
{
	return _starpu_worker_get_local_memory_node();
}

int starpu_memory_node_get_devid(unsigned node)
{
	return _starpu_descr.devid[node];
}

unsigned starpu_memory_devid_find_node(int devid, enum starpu_node_kind kind)
{
	unsigned nnodes = _starpu_memory_nodes_get_count();
	unsigned id;

	for (id = 0; id < nnodes; id++)
	{
		if (_starpu_descr.devid[id] == devid && _starpu_descr.nodes[id]  == kind)
			return id;
	}
	STARPU_ABORT_MSG("can't find node of kind %d and devid %u", kind, devid);
}

enum starpu_worker_archtype starpu_memory_node_get_worker_archtype(enum starpu_node_kind node_kind)
{
	enum starpu_worker_archtype archtype = starpu_memory_driver_info[node_kind].worker_archtype;
	STARPU_ASSERT_MSG(archtype != (enum starpu_worker_archtype) -1, "ambiguous memory node kind %d", node_kind);
	return archtype;
}
