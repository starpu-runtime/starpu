/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015,2017                                CNRS
 * Copyright (C) 2013                                     Université de Bordeaux
 * Copyright (C) 2013                                     Thibaut Lambert
 * Copyright (C) 2012                                     Inria
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

#ifndef __DRIVER_SCC_SOURCE_H__
#define __DRIVER_SCC_SOURCE_H__

#include <starpu.h>
#include <starpu_scc.h>
#include <common/config.h>


#ifdef STARPU_USE_SCC

#include <drivers/mp_common/mp_common.h>
#include <datawizard/node_ops.h>

extern struct _starpu_node_ops _starpu_driver_scc_node_ops;

void _starpu_scc_src_mp_deinit();

void (*_starpu_scc_src_get_kernel_from_job(const struct _starpu_mp_node *,struct _starpu_job *j))(void);
int _starpu_scc_src_register_kernel(starpu_scc_func_symbol_t *symbol, const char *func_name);
starpu_scc_kernel_t _starpu_scc_src_get_kernel(starpu_scc_func_symbol_t symbol);

unsigned _starpu_scc_src_get_device_count();
void _starpu_scc_exit_useless_node(int devid);

void _starpu_scc_src_init(struct _starpu_mp_node *node);

int _starpu_scc_allocate_memory(void **addr, size_t size, unsigned memory_node);
void _starpu_scc_free_memory(void *addr, unsigned memory_node);
int _starpu_scc_allocate_shared_memory(void **addr, size_t size);
void _starpu_scc_free_shared_memory(void *addr);

void _starpu_scc_set_offset_in_shared_memory(void *ptr, void **dev_handle, size_t *offset);

int _starpu_scc_copy_src_to_sink(void *src, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst, unsigned dst_node, size_t size);
int _starpu_scc_copy_sink_to_src(void *src, unsigned src_node, void *dst, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, size_t size);
int _starpu_scc_copy_sink_to_sink(void *src, unsigned src_node, void *dst, unsigned dst_node, size_t size);

void *_starpu_scc_src_worker(void *arg);

int _starpu_scc_copy_data_from_scc_to_cpu(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);
int _starpu_scc_copy_data_from_scc_to_scc(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);
int _starpu_scc_copy_data_from_cpu_to_scc(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);

int _starpu_scc_copy_interface_from_scc_to_cpu(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);
int _starpu_scc_copy_interface_from_scc_to_scc(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);
int _starpu_scc_copy_interface_from_cpu_to_scc(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);

int _starpu_scc_is_direct_access_supported(unsigned node, unsigned handling_node);
uintptr_t _starpu_scc_malloc_on_node(unsigned dst_node, size_t size, int flags);
void _starpu_scc_free_on_node(unsigned dst_node, uintptr_t addr, size_t size, int flags);

#endif /* STARPU_USE_SCC */


#endif /* __DRIVER_SCC_SOURCE_H__ */
