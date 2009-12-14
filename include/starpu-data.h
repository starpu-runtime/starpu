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

#include <starpu_config.h>
#include <starpu-data-interfaces.h>
#include <starpu-data-filters.h>

#ifdef __cplusplus
extern "C" {
#endif

struct starpu_data_state_t;

typedef enum {
	STARPU_R,
	STARPU_W,
	STARPU_RW
} starpu_access_mode;

typedef struct starpu_buffer_descr_t {
	starpu_data_handle handle;
	starpu_access_mode mode;
} starpu_buffer_descr;

void starpu_unpartition_data(struct starpu_data_state_t *root_data, uint32_t gathering_node);
void starpu_delete_data(struct starpu_data_state_t *state);

void starpu_advise_if_data_is_important(struct starpu_data_state_t *state, unsigned is_important);

int starpu_sync_data_with_mem(struct starpu_data_state_t *state);
int starpu_notify_data_modification(struct starpu_data_state_t *state, uint32_t modifying_node);

int starpu_malloc_pinned_if_possible(void **A, size_t dim);
int starpu_free_pinned_if_possible(void *A);

int starpu_request_data_allocation(struct starpu_data_state_t *state, uint32_t node);

int starpu_prefetch_data_on_node(struct starpu_data_state_t *state, unsigned node, unsigned async);

unsigned starpu_get_worker_memory_node(unsigned workerid);

/* It is possible to associate a mask to a piece of data (and its children) so
 * that when it is modified, it is automatically transfered into those memory
 * node. For instance a (1<<0) write-back mask means that the CUDA workers will
 * commit their changes in main memory (node 0). */
void starpu_data_set_wb_mask(struct starpu_data_state_t *state, uint32_t wb_mask);

#ifdef __cplusplus
}
#endif

#endif // __STARPU_DATA_H__
