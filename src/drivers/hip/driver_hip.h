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

#ifndef __DRIVER_HIP_H__
#define __DRIVER_HIP_H__

/** @file */

#include <common/config.h>

void _starpu_hip_preinit(void);

#ifdef STARPU_USE_HIP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wundef"
#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#ifndef __cplusplus
#pragma GCC diagnostic ignored "-Wimplicit-int"
#endif
#pragma GCC diagnostic ignored "-Wreturn-type"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#pragma GCC diagnostic pop
// not needed yet #include <hipblas.h>
#endif

#include <starpu.h>
#include <core/workers.h>
#include <datawizard/node_ops.h>

#pragma GCC visibility push(hidden)

extern struct _starpu_driver_ops _starpu_driver_hip_ops;
extern struct _starpu_node_ops _starpu_driver_hip_node_ops;

extern int _starpu_nworker_per_hip;

void _starpu_hip_init(void);
unsigned _starpu_get_hip_device_count(void);
#ifdef STARPU_HAVE_HWLOC
struct _starpu_machine_topology;
hwloc_obj_t _starpu_hip_get_hwloc_obj(struct _starpu_machine_topology *topology, int devid);
#endif
extern int _starpu_hip_bus_ids[STARPU_MAXHIPDEVS+STARPU_MAXNUMANODES][STARPU_MAXHIPDEVS+STARPU_MAXNUMANODES];

#if defined(STARPU_USE_HIP)
void _starpu_hip_discover_devices(struct _starpu_machine_config *);
void _starpu_init_hip_config(struct _starpu_machine_topology *topology, struct _starpu_machine_config *);
void _starpu_hip_init_worker_binding(struct _starpu_machine_config *config, int no_mp_config, struct _starpu_worker *workerarg);
void _starpu_hip_init_worker_memory(struct _starpu_machine_config *config, int no_mp_config, struct _starpu_worker *workerarg);
void _starpu_init_hip(void);
void *_starpu_hip_worker(void *);
#else
#  define _starpu_hip_discover_devices(config) ((void) config)
#endif

unsigned _starpu_hip_test_request_completion(struct _starpu_async_channel *async_channel);
void _starpu_hip_wait_request_completion(struct _starpu_async_channel *async_channel);

int _starpu_hip_copy_interface_from_cpu_to_hip(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);
int _starpu_hip_copy_interface_from_hip_to_hip(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);
int _starpu_hip_copy_interface_from_hip_to_cpu(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);

int _starpu_hip_copy_data_from_hip_to_hip(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);
int _starpu_hip_copy_data_from_hip_to_cpu(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);
int _starpu_hip_copy_data_from_cpu_to_hip(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);

int _starpu_hip_copy2d_data_from_hip_to_hip(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst, struct _starpu_async_channel *async_channel);
int _starpu_hip_copy2d_data_from_hip_to_cpu(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst, struct _starpu_async_channel *async_channel);
int _starpu_hip_copy2d_data_from_cpu_to_hip(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst, struct _starpu_async_channel *async_channel);

int _starpu_hip_is_direct_access_supported(unsigned node, unsigned handling_node);
uintptr_t _starpu_hip_malloc_on_node(unsigned dst_node, size_t size, int flags);
void _starpu_hip_free_on_node(unsigned dst_node, uintptr_t addr, size_t size, int flags);

#pragma GCC visibility pop

#endif //  __DRIVER_HIP_H__

