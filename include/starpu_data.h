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

#define STARPU_R	(1<<0)
#define STARPU_W	(1<<1)
#define STARPU_RW	(STARPU_R|STARPU_W)
#define STARPU_SCRATCH	(1<<2)
typedef uint32_t starpu_access_mode;

typedef struct starpu_buffer_descr_t {
	starpu_data_handle handle;
	starpu_access_mode mode;
} starpu_buffer_descr;

struct starpu_data_interface_ops_t;

void starpu_data_unregister(starpu_data_handle handle);

/* Destroy all data replicates. After data invalidation, the first access to
 * the handle must be performed in write-only mode. */
void starpu_data_invalidate(starpu_data_handle);

void starpu_data_advise_as_important(starpu_data_handle handle, unsigned is_important);

int starpu_data_acquire(starpu_data_handle handle, starpu_access_mode mode);
int starpu_data_acquire_cb(starpu_data_handle handle,
			starpu_access_mode mode, void (*callback)(void *), void *arg);
void starpu_data_release(starpu_data_handle handle);

int starpu_data_malloc_pinned_if_possible(void **A, size_t dim);
int starpu_data_free_pinned_if_possible(void *A);

int starpu_data_request_allocation(starpu_data_handle handle, uint32_t node);

int starpu_data_prefetch_on_node(starpu_data_handle handle, unsigned node, unsigned async);

unsigned starpu_worker_get_memory_node(unsigned workerid);

/* It is possible to associate a mask to a piece of data (and its children) so
 * that when it is modified, it is automatically transfered into those memory
 * node. For instance a (1<<0) write-back mask means that the CUDA workers will
 * commit their changes in main memory (node 0). */
void starpu_data_set_wb_mask(starpu_data_handle handle, uint32_t wb_mask);

void starpu_data_set_sequential_consistency_flag(starpu_data_handle handle, unsigned flag);
unsigned starpu_data_get_default_sequential_consistency_flag(void);
void starpu_data_set_default_sequential_consistency_flag(unsigned flag);

unsigned starpu_data_test_if_allocated_on_node(starpu_data_handle handle, uint32_t memory_node);

#ifdef __cplusplus
}
#endif

#endif // __STARPU_DATA_H__
