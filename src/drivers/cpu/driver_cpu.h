/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __DRIVER_CPU_H__
#define __DRIVER_CPU_H__

/** @file */

#include <core/workers.h>
#include <common/config.h>
#include <datawizard/node_ops.h>

#pragma GCC visibility push(hidden)

void _starpu_cpu_preinit(void);

extern struct _starpu_driver_ops _starpu_driver_cpu_ops;
extern struct _starpu_node_ops _starpu_driver_cpu_node_ops;

/* Reserve one CPU core as busy for starting a driver thread */
void _starpu_cpu_busy_cpu(unsigned num);

void _starpu_init_cpu_config(struct _starpu_machine_topology *topology, struct _starpu_machine_config *config);
void _starpu_cpu_init_worker_binding(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg);
void _starpu_cpu_init_worker_memory(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg);

void *_starpu_cpu_worker(void *);

int _starpu_cpu_copy_interface(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);
int _starpu_cpu_copy_data(uintptr_t src_ptr, size_t src_offset, int src_dev, uintptr_t dst_ptr, size_t dst_offset, int dst_dev, size_t ssize, struct _starpu_async_channel *async_channel);

int _starpu_cpu_is_direct_access_supported(unsigned node, unsigned handling_node);
uintptr_t _starpu_cpu_malloc_on_device(int dst_node, size_t size, int flags);
void _starpu_cpu_free_on_device(int dst_node, uintptr_t addr, size_t size, int flags);

#pragma GCC visibility pop

#endif //  __DRIVER_CPU_H__
