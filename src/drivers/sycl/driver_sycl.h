/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __DRIVER_SYCL_H__
#define __DRIVER_SYCL_H__

/** @file */

void _starpu_sycl_preinit(void);

#ifdef STARPU_USE_SYCL

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wundef"
#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#ifndef __cplusplus
#pragma GCC diagnostic ignored "-Wimplicit-int"
#endif
#pragma GCC diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic pop
// not needed yet #include <syclblas.h>

#endif

#include <starpu.h>
#include <datawizard/node_ops.h>

#pragma GCC visibility push(hidden)

#ifdef __cplusplus
extern "C" {
#endif

#include <common/config.h>

extern struct _starpu_driver_ops _starpu_driver_sycl_ops;
extern struct _starpu_node_ops _starpu_driver_sycl_node_ops;

extern int _starpu_nworker_per_sycl;

void _starpu_sycl_early_init(void);
#ifdef STARPU_HAVE_HWLOC
struct _starpu_machine_topology;
hwloc_obj_t _starpu_sycl_get_hwloc_obj(hwloc_topology_t topology, int devid);
#endif
extern int _starpu_sycl_bus_ids[STARPU_MAXSYCLDEVS+STARPU_MAXNUMANODES][STARPU_MAXSYCLDEVS+STARPU_MAXNUMANODES];

#if defined(STARPU_USE_SYCL)
void _starpu_sycl_discover_devices (struct _starpu_machine_config *);
void _starpu_init_sycl_config(struct _starpu_machine_topology *topology, struct _starpu_machine_config *, int no_mp_config);
void _starpu_sycl_init_worker_binding(struct _starpu_machine_config *config, int no_mp_config, struct _starpu_worker *workerarg);
void _starpu_sycl_init_worker_memory(struct _starpu_machine_config *config, int no_mp_config, struct _starpu_worker *workerarg);
void _starpu_init_sycl(void);
void *_starpu_sycl_worker(void *);
#else
#  define _starpu_sycl_discover_devices(config) ((void) config)
#endif

unsigned _starpu_sycl_test_request_completion(struct _starpu_async_channel *async_channel);
void _starpu_sycl_wait_request_completion(struct _starpu_async_channel *async_channel);

int _starpu_sycl_copy_interface_from_cpu_to_sycl(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);
int _starpu_sycl_copy_interface_from_sycl_to_sycl(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);
int _starpu_sycl_copy_interface_from_sycl_to_cpu(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);

int _starpu_sycl_copy_data_from_sycl_to_sycl(uintptr_t src, size_t src_offset, int src_devid, uintptr_t dst, size_t dst_offset, int dst_devid, size_t size, struct _starpu_async_channel *async_channel);
int _starpu_sycl_copy_data_from_sycl_to_cpu(uintptr_t src, size_t src_offset, int src_devid, uintptr_t dst, size_t dst_offset, int dst_devid, size_t size, struct _starpu_async_channel *async_channel);
int _starpu_sycl_copy_data_from_cpu_to_sycl(uintptr_t src, size_t src_offset, int src_devid, uintptr_t dst, size_t dst_offset, int dst_devid, size_t size, struct _starpu_async_channel *async_channel);

int _starpu_sycl_copy2d_data_from_sycl_to_sycl(uintptr_t src, size_t src_offset, int src_devid, uintptr_t dst, size_t dst_offset, int dst_devid, size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst, struct _starpu_async_channel *async_channel);
int _starpu_sycl_copy2d_data_from_sycl_to_cpu(uintptr_t src, size_t src_offset, int src_devid, uintptr_t dst, size_t dst_offset, int dst_devid, size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst, struct _starpu_async_channel *async_channel);
int _starpu_sycl_copy2d_data_from_cpu_to_sycl(uintptr_t src, size_t src_offset, int src_devid, uintptr_t dst, size_t dst_offset, int dst_devid, size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst, struct _starpu_async_channel *async_channel);

int _starpu_sycl_is_direct_access_supported(unsigned node, unsigned handling_node);
uintptr_t _starpu_sycl_malloc_on_dev(int devid, size_t size, int flags);
void _starpu_sycl_free_on_dev(int devid, uintptr_t addr, size_t size, int flags);
void _starpu_sycl_device_name(int devid, char *name, size_t size);
size_t _starpu_sycl_total_memory(int devid);
void _starpu_sycl_init_device_context(int devid);
void _starpu_sycl_reset_device(int devid);
int _starpu_sycl_peer_access(int devid, int peer_devid);

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop

#endif //  __DRIVER_SYCL_H__

