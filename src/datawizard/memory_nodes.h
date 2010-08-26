/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
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

#ifndef __MEMORY_NODES_H__
#define __MEMORY_NODES_H__

#include <starpu.h>
#include <common/config.h>
#include <datawizard/coherency.h>
#include <datawizard/memalloc.h>

typedef enum {
	STARPU_UNUSED     = 0x00,
	STARPU_CPU_RAM    = 0x01,
	STARPU_CUDA_RAM   = 0x02,
	STARPU_OPENCL_RAM = 0x03,
	STARPU_SPU_LS     = 0x04
} starpu_node_kind;

typedef starpu_node_kind starpu_memory_node_tuple;

#define _STARPU_MEMORY_NODE_TUPLE(node1,node2) (node1 | (node2 << 4))
#define _STARPU_MEMORY_NODE_TUPLE_FIRST(tuple) (tuple & 0x0F)
#define _STARPU_MEMORY_NODE_TUPLE_SECOND(tuple) (tuple & 0xF0)

struct _cond_and_mutex {
        pthread_cond_t *cond;
        pthread_mutex_t *mutex;	
};

typedef struct {
	unsigned nnodes;
	starpu_node_kind nodes[STARPU_MAXNODES];

	// TODO move this 2 lists outside starpu_mem_node_descr
	/* Every worker is associated to a condition variable on which the
	 * worker waits when there is task available. It is possible that
	 * multiple worker share the same condition variable, so we maintain a
	 * list of all these condition variables so that we can wake up all
	 * worker attached to a memory node that are waiting on a task. */
	pthread_rwlock_t conditions_rwlock;
	struct _cond_and_mutex conditions_attached_to_node[STARPU_MAXNODES][STARPU_NMAXWORKERS];
	struct _cond_and_mutex conditions_all[STARPU_MAXNODES*STARPU_NMAXWORKERS];
	/* the number of queues attached to each node */
	unsigned total_condition_count;
	unsigned condition_count[STARPU_MAXNODES];

} starpu_mem_node_descr;

void _starpu_init_memory_nodes(void);
void _starpu_deinit_memory_nodes(void);
void _starpu_set_local_memory_node_key(unsigned *node);
unsigned _starpu_get_local_memory_node(void);
unsigned _starpu_register_memory_node(starpu_node_kind kind);
//void _starpu_memory_node_attach_queue(struct starpu_jobq_s *q, unsigned nodeid);
void _starpu_memory_node_register_condition(pthread_cond_t *cond, pthread_mutex_t *mutex, unsigned memory_node);

starpu_node_kind _starpu_get_node_kind(uint32_t node);
unsigned _starpu_get_memory_nodes_count(void);

starpu_mem_node_descr *_starpu_get_memory_node_description(void);

#endif // __MEMORY_NODES_H__
