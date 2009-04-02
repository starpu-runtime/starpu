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

#ifndef __COPY_DRIVER_H__
#define __COPY_DRIVER_H__

#include "coherency.h"
#include "memalloc.h"

#ifdef USE_CUDA
#include <cublas.h>
#endif


typedef enum {
	UNUSED,
	SPU_LS,
	RAM,
	CUDA_RAM
} node_kind;

typedef struct {
	unsigned nnodes;
	node_kind nodes[MAXNODES];

	/* the list of queues that are attached to a given node */
	// XXX 32 is set randomly !
	struct jobq_s *attached_queues[MAXNODES][32];
	/* the number of queues attached to each node */
	unsigned queues_count[MAXNODES];
} mem_node_descr;

struct starpu_data_state_t;

__attribute__((warn_unused_result))
int driver_copy_data(struct starpu_data_state_t *state, uint32_t src_node_mask, uint32_t dst_node, unsigned donotread);

void init_memory_nodes(void);
void set_local_memory_node_key(unsigned *node);
unsigned get_local_memory_node(void);
unsigned register_memory_node(node_kind kind);
void memory_node_attach_queue(struct jobq_s *q, unsigned nodeid);
void wake_all_blocked_workers(void);
void wake_all_blocked_workers_on_node(unsigned nodeid);

node_kind get_node_kind(uint32_t node);

__attribute__((warn_unused_result))
int driver_copy_data_1_to_1(struct starpu_data_state_t *state, uint32_t node, 
				uint32_t requesting_node, unsigned donotread);

int allocate_per_node_buffer(struct starpu_data_state_t *state, uint32_t node);

#endif // __COPY_DRIVER_H__
