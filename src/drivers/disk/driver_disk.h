/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Corentin Salingue
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

#ifndef __DRIVER_DISK_H__
#define __DRIVER_DISK_H__

/** @file */

#include <datawizard/node_ops.h>

int _starpu_disk_copy_src_to_disk(void * src, unsigned src_node, void * dst, size_t dst_offset, unsigned dst_node, size_t size, void * async_channel);

int _starpu_disk_copy_disk_to_src(void * src, size_t src_offset, unsigned src_node, void * dst, unsigned dst_node, size_t size, void * async_channel);

int _starpu_disk_copy_disk_to_disk(void * src, size_t src_offset, unsigned src_node, void * dst, size_t dst_offset, unsigned dst_node, size_t size, void * async_channel);

unsigned _starpu_disk_test_request_completion(struct _starpu_async_channel *async_channel);
void _starpu_disk_wait_request_completion(struct _starpu_async_channel *async_channel);

int _starpu_disk_copy_interface_from_disk_to_cpu(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);
int _starpu_disk_copy_interface_from_disk_to_disk(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);
int _starpu_disk_copy_interface_from_cpu_to_disk(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);

int _starpu_disk_copy_data_from_disk_to_cpu(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);
int _starpu_disk_copy_data_from_disk_to_disk(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);
int _starpu_disk_copy_data_from_cpu_to_disk(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);

extern struct _starpu_node_ops _starpu_driver_disk_node_ops;
int _starpu_disk_is_direct_access_supported(unsigned node, unsigned handling_node);
uintptr_t _starpu_disk_malloc_on_node(unsigned dst_node, size_t size, int flags);
void _starpu_disk_free_on_node(unsigned dst_node, uintptr_t addr, size_t size, int flags);

#endif
