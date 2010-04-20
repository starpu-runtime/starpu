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

#ifndef __STARPU_DATA_H__
#define __STARPU_DATA_H__

#include <starpu.h>
#include <starpu_config.h>

struct starpu_data_state_t;
typedef struct starpu_data_state_t * starpu_data_handle;

#include <starpu_data_interfaces.h>
#include <starpu_data_filters.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
	STARPU_R,
	STARPU_W,
	STARPU_RW
} starpu_access_mode;

typedef struct starpu_buffer_descr_t {
	starpu_data_handle handle;
	starpu_access_mode mode;
} starpu_buffer_descr;

struct starpu_data_interface_ops_t;

void starpu_unpartition_data(starpu_data_handle root_data, uint32_t gathering_node);
void starpu_delete_data(starpu_data_handle state);

void starpu_advise_if_data_is_important(starpu_data_handle state, unsigned is_important);

int starpu_sync_data_with_mem(starpu_data_handle state, starpu_access_mode mode);
int starpu_sync_data_with_mem_non_blocking(starpu_data_handle handle,
			starpu_access_mode mode, void (*callback)(void *), void *arg);
void starpu_release_data_from_mem(starpu_data_handle state);

int starpu_malloc_pinned_if_possible(void **A, size_t dim);
int starpu_free_pinned_if_possible(void *A);

int starpu_request_data_allocation(starpu_data_handle state, uint32_t node);

int starpu_prefetch_data_on_node(starpu_data_handle state, unsigned node, unsigned async);

unsigned starpu_get_worker_memory_node(unsigned workerid);

/* It is possible to associate a mask to a piece of data (and its children) so
 * that when it is modified, it is automatically transfered into those memory
 * node. For instance a (1<<0) write-back mask means that the CUDA workers will
 * commit their changes in main memory (node 0). */
void starpu_data_set_wb_mask(starpu_data_handle state, uint32_t wb_mask);

unsigned starpu_test_if_data_is_allocated_on_node(starpu_data_handle handle, uint32_t memory_node);

#ifdef __cplusplus
}
#endif

#endif // __STARPU_DATA_H__
