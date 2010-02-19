/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <pthread.h>
#include <common/config.h>
#include <core/policies/sched_policy.h>
#include <datawizard/datastats.h>
#include <common/fxt.h>
#include "copy-driver.h"
#include "memalloc.h"

static starpu_mem_node_descr descr;
static pthread_key_t memory_node_key;

void starpu_init_memory_nodes(void)
{
	/* there is no node yet, subsequent nodes will be 
	 * added using _starpu_register_memory_node */
	descr.nnodes = 0;

	pthread_key_create(&memory_node_key, NULL);

	unsigned i;
	for (i = 0; i < STARPU_MAXNODES; i++) 
		descr.nodes[i] = STARPU_UNUSED; 

	_starpu_init_mem_chunk_lists();
	starpu_init_data_request_lists();

	pthread_rwlock_init(&descr.attached_queues_rwlock, NULL);
	descr.total_queues_count = 0;
}

void starpu_deinit_memory_nodes(void)
{
	starpu_deinit_data_request_lists();
	_starpu_deinit_mem_chunk_lists();

	pthread_key_delete(memory_node_key);
}

void starpu_set_local_memory_node_key(unsigned *node)
{
	pthread_setspecific(memory_node_key, node);
}

unsigned starpu_get_local_memory_node(void)
{
	unsigned *memory_node;
	memory_node = pthread_getspecific(memory_node_key);
	
	/* in case this is called by the programmer, we assume the STARPU_RAM node 
	   is the appropriate memory node ... so we return 0 XXX */
	if (STARPU_UNLIKELY(!memory_node))
		return 0;

	return *memory_node;
}

inline starpu_mem_node_descr *starpu_get_memory_node_description(void)
{
	return &descr;
}

inline starpu_node_kind starpu_get_node_kind(uint32_t node)
{
	return descr.nodes[node];
}

unsigned starpu_get_memory_nodes_count(void)
{
	return descr.nnodes;
}

unsigned _starpu_register_memory_node(starpu_node_kind kind)
{
	unsigned nnodes;
	/* ATOMIC_ADD returns the new value ... */
	nnodes = STARPU_ATOMIC_ADD(&descr.nnodes, 1);

	descr.nodes[nnodes-1] = kind;
	TRACE_NEW_MEM_NODE(nnodes-1);

	/* for now, there is no queue related to that newly created node */
	descr.queues_count[nnodes-1] = 0;

	return (nnodes-1);
}

/* TODO move in a more appropriate file  !! */
/* attach a queue to a memory node (if it's not already attached) */
void starpu_memory_node_attach_queue(struct jobq_s *q, unsigned nodeid)
{
	unsigned queue;
	unsigned nqueues_total, nqueues;
	
	pthread_rwlock_wrlock(&descr.attached_queues_rwlock);

	/* we only insert the queue if it's not already in the list */
	nqueues = descr.queues_count[nodeid];
	for (queue = 0; queue < nqueues; queue++)
	{
		if (descr.attached_queues_per_node[nodeid][queue] == q)
		{
			/* the queue is already in the list */
			pthread_rwlock_unlock(&descr.attached_queues_rwlock);
			return;
		}
	}

	/* it was not found locally */
	descr.attached_queues_per_node[nodeid][nqueues] = q;
	descr.queues_count[nodeid]++;

	/* do we have to add it in the global list as well ? */
	nqueues_total = descr.total_queues_count; 
	for (queue = 0; queue < nqueues_total; queue++)
	{
		if (descr.attached_queues_all[queue] == q)
		{
			/* the queue is already in the global list */
			pthread_rwlock_unlock(&descr.attached_queues_rwlock);
			return;
		}
	} 

	/* it was not in the global queue either */
	descr.attached_queues_all[nqueues_total] = q;
	descr.total_queues_count++;

	pthread_rwlock_unlock(&descr.attached_queues_rwlock);
}

unsigned starpu_get_worker_memory_node(unsigned workerid)
{
	struct worker_s *worker = _starpu_get_worker_struct(workerid);

	return worker->memory_node;
}
