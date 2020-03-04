/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __DRIVER_OPENCL_H__
#define __DRIVER_OPENCL_H__

/** @file */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#ifdef STARPU_USE_OPENCL

#define CL_TARGET_OPENCL_VERSION 100
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#endif

#include <core/workers.h>
#include <datawizard/node_ops.h>

#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
struct _starpu_machine_config;
void _starpu_opencl_discover_devices(struct _starpu_machine_config *config);

unsigned _starpu_opencl_get_device_count(void);
void _starpu_opencl_init(void);
void *_starpu_opencl_worker(void *);
extern struct _starpu_node_ops _starpu_driver_opencl_node_ops;
#else
#define _starpu_opencl_discover_devices(config) ((void) (config))
#endif

#ifdef STARPU_USE_OPENCL
extern struct _starpu_driver_ops _starpu_driver_opencl_ops;
extern char *_starpu_opencl_program_dir;

int _starpu_run_opencl(struct _starpu_worker *);
int _starpu_opencl_driver_init(struct _starpu_worker *);
int _starpu_opencl_driver_run_once(struct _starpu_worker *);
int _starpu_opencl_driver_deinit(struct _starpu_worker *);

int _starpu_opencl_init_context(int devid);
int _starpu_opencl_deinit_context(int devid);
cl_device_type _starpu_opencl_get_device_type(int devid);
#endif

#if 0
cl_int _starpu_opencl_copy_rect_opencl_to_ram(cl_mem buffer, unsigned src_node, void *ptr, unsigned dst_node, const size_t buffer_origin[3], const size_t host_origin[3],
                                              const size_t region[3], size_t buffer_row_pitch, size_t buffer_slice_pitch,
                                              size_t host_row_pitch, size_t host_slice_pitch, cl_event *event);

cl_int _starpu_opencl_copy_rect_ram_to_opencl(void *ptr, unsigned src_node, cl_mem buffer, unsigned dst_node, const size_t buffer_origin[3], const size_t host_origin[3],
                                              const size_t region[3], size_t buffer_row_pitch, size_t buffer_slice_pitch,
                                              size_t host_row_pitch, size_t host_slice_pitch, cl_event *event);
#endif

unsigned _starpu_opencl_test_request_completion(struct _starpu_async_channel *async_channel);
void _starpu_opencl_wait_request_completion(struct _starpu_async_channel *async_channel);

int _starpu_opencl_copy_interface_from_opencl_to_opencl(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);
int _starpu_opencl_copy_interface_from_opencl_to_cpu(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);
int _starpu_opencl_copy_interface_from_cpu_to_opencl(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);

int _starpu_opencl_copy_data_from_opencl_to_cpu(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);
int _starpu_opencl_copy_data_from_opencl_to_opencl(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);
int _starpu_opencl_copy_data_from_cpu_to_opencl(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);

int _starpu_opencl_is_direct_access_supported(unsigned node, unsigned handling_node);
uintptr_t _starpu_opencl_malloc_on_node(unsigned dst_node, size_t size, int flags);
void _starpu_opencl_free_on_node(unsigned dst_node, uintptr_t addr, size_t size, int flags);

#endif //  __DRIVER_OPENCL_H__
