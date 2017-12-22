/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012  Inria
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

#endif /* STARPU_USE_SCC */


#endif /* __DRIVER_SCC_SOURCE_H__ */
