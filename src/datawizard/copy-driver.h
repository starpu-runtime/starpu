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

#include <common/config.h>
#include <datawizard/memory_nodes.h>
#include "coherency.h"
#include "memalloc.h"

#ifdef USE_CUDA
#include <cublas.h>
#endif

struct starpu_data_state_t;

struct copy_data_methods_s {
	/* src type is ram */
	int (*ram_to_ram)(struct starpu_data_state_t *state, uint32_t src, uint32_t dst);
	int (*ram_to_cuda)(struct starpu_data_state_t *state, uint32_t src, uint32_t dst);
	int (*ram_to_spu)(struct starpu_data_state_t *state, uint32_t src, uint32_t dst);

	/* src type is cuda */
	int (*cuda_to_ram)(struct starpu_data_state_t *state, uint32_t src, uint32_t dst);
	int (*cuda_to_cuda)(struct starpu_data_state_t *state, uint32_t src, uint32_t dst);
	int (*cuda_to_spu)(struct starpu_data_state_t *state, uint32_t src, uint32_t dst);

	/* src type is spu */
	int (*spu_to_ram)(struct starpu_data_state_t *state, uint32_t src, uint32_t dst);
	int (*spu_to_cuda)(struct starpu_data_state_t *state, uint32_t src, uint32_t dst);
	int (*spu_to_spu)(struct starpu_data_state_t *state, uint32_t src, uint32_t dst);
};

void wake_all_blocked_workers(void);
void wake_all_blocked_workers_on_node(unsigned nodeid);

__attribute__((warn_unused_result))
int driver_copy_data_1_to_1(struct starpu_data_state_t *state, uint32_t node, 
				uint32_t requesting_node, unsigned donotread);

int allocate_per_node_buffer(struct starpu_data_state_t *state, uint32_t node);

#endif // __COPY_DRIVER_H__
