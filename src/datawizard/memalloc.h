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

#ifndef __MEMALLOC_H__
#define __MEMALLOC_H__

#include <starpu.h>
#include <common/config.h>

#include <common/list.h>
#include <datawizard/interfaces/data_interface.h>
#include <datawizard/coherency.h>
#include <datawizard/copy_driver.h>

LIST_TYPE(starpu_mem_chunk,
	starpu_data_handle data;
	size_t size;

	uint32_t footprint;
	
	/* The footprint of the data is not sufficient to determine whether two
	 * pieces of data have the same layout (there could be collision in the
	 * hash function ...) so we still keep a copy of the actual layout (ie.
	 * the data interface) to stay on the safe side. We make a copy of
	 * because when a data is deleted, the memory chunk remains.
	 */
	struct starpu_data_interface_ops_t *ops;
	void *interface;
	unsigned automatically_allocated;
	unsigned data_was_deleted;
)

void _starpu_init_mem_chunk_lists(void);
void _starpu_deinit_mem_chunk_lists(void);
void _starpu_request_mem_chunk_removal(starpu_data_handle handle, unsigned node);
ssize_t _starpu_allocate_interface(starpu_data_handle handle, void *interface, uint32_t dst_node);
int _starpu_allocate_memory_on_node(starpu_data_handle handle, uint32_t dst_node, unsigned may_alloc);
size_t _starpu_free_all_automatically_allocated_buffers(uint32_t node);

/* Memory chunk cache */
void _starpu_memchunk_cache_insert(uint32_t node, starpu_mem_chunk_t);
starpu_mem_chunk_t _starpu_memchunk_cache_lookup(uint32_t node, starpu_data_handle handle);
starpu_mem_chunk_t _starpu_memchunk_init(starpu_data_handle handle, size_t size, void *interface,
			size_t interface_size, unsigned automatically_allocated);
#endif
