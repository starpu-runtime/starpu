/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2010, 2012-2015  Université de Bordeaux
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

#ifndef __MEMALLOC_H__
#define __MEMALLOC_H__

#include <starpu.h>
#include <common/config.h>

#include <common/list.h>
#include <datawizard/interfaces/data_interface.h>
#include <datawizard/coherency.h>
#include <datawizard/copy_driver.h>

struct _starpu_data_replicate;

/* While associated with a handle, the content is protected by the handle lock, except a few fields
 */
LIST_TYPE(_starpu_mem_chunk,
	/* protected by the mc_lock */
	starpu_data_handle_t data;

	uint32_t footprint;

	/*
	 * When re-using a memchunk, the footprint of the data is not
	 * sufficient to determine whether two pieces of data have the same
	 * layout (there could be collision in the hash function ...) so we
	 * still keep a copy of the actual layout (ie. the data interface) to
	 * stay on the safe side while the memchunk is detached from an actual
	 * data.
	 */
	struct starpu_data_interface_ops *ops;
	void *chunk_interface;
	size_t size_interface;
	unsigned automatically_allocated;

	/* the size of the data is only set when calling _starpu_request_mem_chunk_removal(),
         * it is needed by free_memory_on_node() which is called when
         * the handle is no longer valid. It should not be used otherwise.
	 */
	size_t size;

	/* A buffer that is used for SCRATCH or reduction cannnot be used with
	 * filters. */
	unsigned relaxed_coherency;
	struct _starpu_data_replicate *replicate;

	/* This is set when one keeps a pointer to this mc obtained from the
	 * mc_list without mc_lock held. We need to clear the pointer if we
	 * remove this entry from the mc_list, so we know we have to restart
	 * from zero. This is protected by the corresponding mc_lock.  */
	struct _starpu_mem_chunk **remove_notify;
)

/* LRU list */
LIST_TYPE(_starpu_mem_chunk_lru,
	struct _starpu_mem_chunk *mc;
)

void _starpu_init_mem_chunk_lists(void);
void _starpu_deinit_mem_chunk_lists(void);
void _starpu_request_mem_chunk_removal(starpu_data_handle_t handle, struct _starpu_data_replicate *replicate, unsigned node, size_t size);
int _starpu_allocate_memory_on_node(starpu_data_handle_t handle, struct _starpu_data_replicate *replicate, unsigned is_prefetch);
size_t _starpu_free_all_automatically_allocated_buffers(unsigned node);
void _starpu_memchunk_recently_used(struct _starpu_mem_chunk *mc, unsigned node);

void _starpu_display_memory_stats_by_node(int node);
size_t _starpu_memory_reclaim_generic(unsigned node, unsigned force, size_t reclaim);
int _starpu_is_reclaiming(unsigned node);

#endif
