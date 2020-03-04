/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __NODE_OPS_H__
#define __NODE_OPS_H__

/** @file */

#include <starpu.h>
#include <common/config.h>
#include <datawizard/copy_driver.h>

typedef int (*copy_interface_func_t)(starpu_data_handle_t handle, void *src_interface, unsigned src_node,
				void *dst_interface, unsigned dst_node,
				struct _starpu_data_request *req);

typedef int (*copy_data_t)(uintptr_t src_ptr, size_t src_offset, unsigned src_node,
				uintptr_t dst_ptr, size_t dst_offset, unsigned dst_node,
				size_t ssize, struct _starpu_async_channel *async_channel);

typedef int (*copy2d_data_t)(uintptr_t src_ptr, size_t src_offset, unsigned src_node,
				uintptr_t dst_ptr, size_t dst_offset, unsigned dst_node,
				size_t blocksize,
				size_t numblocks, size_t ld_src, size_t ld_dst,
				struct _starpu_async_channel *async_channel);

typedef int (*copy3d_data_t)(uintptr_t src_ptr, size_t src_offset, unsigned src_node,
				uintptr_t dst_ptr, size_t dst_offset, unsigned dst_node,
				size_t blocksize,
				size_t numblocks_1, size_t ld1_src, size_t ld1_dst,
				size_t numblocks_2, size_t ld2_src, size_t ld2_dst,
				struct _starpu_async_channel *async_channel);

struct _starpu_node_ops
{
	copy_interface_func_t copy_interface_to[STARPU_MPI_MS_RAM+1];
	copy_data_t copy_data_to[STARPU_MPI_MS_RAM+1];
	copy2d_data_t copy2d_data_to[STARPU_MPI_MS_RAM+1];
	copy3d_data_t copy3d_data_to[STARPU_MPI_MS_RAM+1];
	void (*wait_request_completion)(struct _starpu_async_channel *async_channel);
	unsigned (*test_request_completion)(struct _starpu_async_channel *async_channel);
	int (*is_direct_access_supported)(unsigned node, unsigned handling_node);
	uintptr_t (*malloc_on_node)(unsigned dst_node, size_t size, int flags);
	void (*free_on_node)(unsigned dst_node, uintptr_t addr, size_t size, int flags);
	char *name;
};

const char* _starpu_node_get_prefix(enum starpu_node_kind kind);

#endif // __NODE_OPS_H__
