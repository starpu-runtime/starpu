/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#pragma GCC visibility push(hidden)

void _starpu_opencl_preinit(void);

#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
struct _starpu_machine_config;
void _starpu_opencl_discover_devices(struct _starpu_machine_config *config);

void _starpu_opencl_early_init(void);
void _starpu_opencl_init_driver(struct _starpu_machine_config *config);
unsigned _starpu_opencl_get_device_count(void);
#ifdef STARPU_HAVE_HWLOC
struct _starpu_machine_topology;
hwloc_obj_t _starpu_opencl_get_hwloc_obj(hwloc_topology_t topology, int devid);
#endif
void _starpu_init_opencl_config(struct _starpu_machine_topology *topology, struct _starpu_machine_config *);
void _starpu_opencl_init_worker_binding(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg);
void _starpu_opencl_init_worker_memory(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED, struct _starpu_worker *workerarg);
void *_starpu_opencl_worker(void *);
extern struct _starpu_node_ops _starpu_driver_opencl_node_ops;
#else
#define _starpu_opencl_discover_devices(config) ((void) (config))
#endif

#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
extern struct _starpu_driver_ops _starpu_driver_opencl_ops;
#endif

#ifdef STARPU_USE_OPENCL
extern char *_starpu_opencl_program_dir;

cl_device_type _starpu_opencl_get_device_type(int devid);
#endif

#pragma GCC visibility pop

#endif //  __DRIVER_OPENCL_H__
