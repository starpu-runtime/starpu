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

#ifndef __MEMORY_NODES_H__
#define __MEMORY_NODES_H__

#include "coherency.h"
#include "memalloc.h"

#ifdef STARPU_USE_CUDA
#include <cublas.h>
#endif

typedef enum {
	STARPU_UNUSED,
	STARPU_SPU_LS,
	STARPU_RAM,
	STARPU_CUDA_RAM
} starpu_node_kind;

typedef struct {
	unsigned nnodes;
	starpu_node_kind nodes[STARPU_MAXNODES];

	/* the list of queues that are attached to a given node */
	// XXX 32 is set randomly !
	// TODO move this 2 lists outside starpu_mem_node_descr
	pthread_rwlock_t attached_queues_rwlock;
	struct starpu_jobq_s *attached_queues_per_node[STARPU_MAXNODES][32];
	struct starpu_jobq_s *attached_queues_all[STARPU_MAXNODES*32];
	/* the number of queues attached to each node */
	unsigned total_queues_count;
	unsigned queues_count[STARPU_MAXNODES];
} starpu_mem_node_descr;

void _starpu_init_memory_nodes(void);
void _starpu_deinit_memory_nodes(void);
void _starpu_set_local_memory_node_key(unsigned *node);
unsigned _starpu_get_local_memory_node(void);
unsigned _starpu_register_memory_node(starpu_node_kind kind);
void _starpu_memory_node_attach_queue(struct starpu_jobq_s *q, unsigned nodeid);

starpu_node_kind _starpu_get_node_kind(uint32_t node);
unsigned _starpu_get_memory_nodes_count(void);

inline starpu_mem_node_descr *_starpu_get_memory_node_description(void);

#endif // __MEMORY_NODES_H__
