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

#ifndef __MEMALLOC_H__
#define __MEMALLOC_H__

#include <common/list.h>
#include <datawizard/interfaces/data_interface.h>
#include <datawizard/coherency.h>
#include <datawizard/copy-driver.h>
#include <datawizard/progress.h>

struct starpu_data_state_t;

LIST_TYPE(mem_chunk,
	struct starpu_data_state_t *data;
	size_t size;

	uint32_t footprint;
	
	/* The footprint of the data is not sufficient to determine whether two
	 * pieces of data have the same layout (there could be collision in the
	 * hash function ...) so we still keep a copy of the actual layout (ie.
	 * the starpu_data_interface_t) to stay on the safe side. We make a copy of
	 * because when a data is deleted, the memory chunk remains.
	 */
	struct data_interface_ops_t *ops;
	void *interface;
	unsigned automatically_allocated;
	unsigned data_was_deleted;
)

void init_mem_chunk_lists(void);
void deinit_mem_chunk_lists(void);
void request_mem_chunk_removal(struct starpu_data_state_t *state, unsigned node);
int allocate_memory_on_node(struct starpu_data_state_t *state, uint32_t dst_node, unsigned may_alloc);

#endif
