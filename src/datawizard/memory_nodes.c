/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
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
#include <common/fxt.h>
#include "copy_driver.h"
#include "memalloc.h"

static struct _starpu_memory_node_descr descr;
static starpu_pthread_key_t memory_node_key;

void _starpu_memory_nodes_init(void)
{
	/* there is no node yet, subsequent nodes will be
	 * added using _starpu_memory_node_register */
	descr.nnodes = 0;

	STARPU_PTHREAD_KEY_CREATE(&memory_node_key, NULL);

	unsigned i;
	for (i = 0; i < STARPU_MAXNODES; i++)
	{
		descr.nodes[i] = STARPU_UNUSED;
		descr.nworkers[i] = 0;
	}

	_starpu_init_mem_chunk_lists();
	_starpu_init_data_request_lists();
	_starpu_memory_manager_init();

	STARPU_PTHREAD_RWLOCK_INIT(&descr.conditions_rwlock, NULL);
	descr.total_condition_count = 0;
}

void _starpu_memory_nodes_deinit(void)
{
	_starpu_deinit_data_request_lists();
	_starpu_deinit_mem_chunk_lists();

	STARPU_PTHREAD_RWLOCK_DESTROY(&descr.conditions_rwlock);
	STARPU_PTHREAD_KEY_DELETE(memory_node_key);
}

void _starpu_memory_node_set_local_key(unsigned *node)
{
	STARPU_PTHREAD_SETSPECIFIC(memory_node_key, node);
}

unsigned _starpu_memory_node_get_local_key(void)
{
	unsigned *memory_node;
	memory_node = (unsigned *) STARPU_PTHREAD_GETSPECIFIC(memory_node_key);

	/* in case this is called by the programmer, we assume the RAM node
	   is the appropriate memory node ... so we return 0 XXX */
	if (STARPU_UNLIKELY(!memory_node))
		return 0;

	return *memory_node;
}

void _starpu_memory_node_add_nworkers(unsigned node)
{
	descr.nworkers[node]++;
}

unsigned _starpu_memory_node_get_nworkers(unsigned node)
{
	return descr.nworkers[node];
}

struct _starpu_memory_node_descr *_starpu_memory_node_get_description(void)
{
	return &descr;
}

enum starpu_node_kind starpu_node_get_kind(unsigned node)
{
	return descr.nodes[node];
}

int _starpu_memory_node_get_devid(unsigned node)
{
	return descr.devid[node];
}

unsigned starpu_memory_nodes_get_count(void)
{
	return descr.nnodes;
}

unsigned _starpu_memory_node_register(enum starpu_node_kind kind, int devid)
{
	unsigned nnodes;
	/* ATOMIC_ADD returns the new value ... */
	nnodes = STARPU_ATOMIC_ADD(&descr.nnodes, 1);
	STARPU_ASSERT_MSG(nnodes < STARPU_MAXNODES,"Too many nodes !");

	descr.nodes[nnodes-1] = kind;
	_STARPU_TRACE_NEW_MEM_NODE(nnodes-1);

	descr.devid[nnodes-1] = devid;

	/* for now, there is no condition associated to that newly created node */
	descr.condition_count[nnodes-1] = 0;

	return (nnodes-1);

}

#ifdef STARPU_SIMGRID
void _starpu_simgrid_memory_node_set_host(unsigned node, msg_host_t host)
{
	descr.host[node] = host;
}

msg_host_t _starpu_simgrid_memory_node_get_host(unsigned node)
{
	return descr.host[node];
}
#endif

/* TODO move in a more appropriate file  !! */
/* Register a condition variable associated to worker which is associated to a
 * memory node itself. */
void _starpu_memory_node_register_condition(starpu_pthread_cond_t *cond, starpu_pthread_mutex_t *mutex, unsigned nodeid)
{
	unsigned cond_id;
	unsigned nconds_total, nconds;

	STARPU_PTHREAD_RWLOCK_WRLOCK(&descr.conditions_rwlock);

	/* we only insert the queue if it's not already in the list */
	nconds = descr.condition_count[nodeid];
	for (cond_id = 0; cond_id < nconds; cond_id++)
	{
		if (descr.conditions_attached_to_node[nodeid][cond_id].cond == cond)
		{
			STARPU_ASSERT(descr.conditions_attached_to_node[nodeid][cond_id].mutex == mutex);

			/* the condition is already in the list */
			STARPU_PTHREAD_RWLOCK_UNLOCK(&descr.conditions_rwlock);
			return;
		}
	}

	/* it was not found locally */
	descr.conditions_attached_to_node[nodeid][cond_id].cond = cond;
	descr.conditions_attached_to_node[nodeid][cond_id].mutex = mutex;
	descr.condition_count[nodeid]++;

	/* do we have to add it in the global list as well ? */
	nconds_total = descr.total_condition_count;
	for (cond_id = 0; cond_id < nconds_total; cond_id++)
	{
		if (descr.conditions_all[cond_id].cond == cond)
		{
			/* the queue is already in the global list */
			STARPU_PTHREAD_RWLOCK_UNLOCK(&descr.conditions_rwlock);
			return;
		}
	}

	/* it was not in the global list either */
	descr.conditions_all[nconds_total].cond = cond;
	descr.conditions_all[nconds_total].mutex = mutex;
	descr.total_condition_count++;

	STARPU_PTHREAD_RWLOCK_UNLOCK(&descr.conditions_rwlock);
}

unsigned starpu_worker_get_memory_node(unsigned workerid)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();

	/* This workerid may either be a basic worker or a combined worker */
	unsigned nworkers = config->topology.nworkers;

	if (workerid < config->topology.nworkers)
		return config->workers[workerid].memory_node;

	/* We have a combined worker */
	unsigned ncombinedworkers = config->topology.ncombinedworkers;
	STARPU_ASSERT_MSG(workerid < ncombinedworkers + nworkers, "Bad workerid %u, maximum %u", workerid, ncombinedworkers + nworkers);
	return config->combined_workers[workerid - nworkers].memory_node;

}
