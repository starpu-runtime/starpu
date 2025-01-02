/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#ifdef STARPU_HAVE_HWLOC
struct _starpu_machine_topology;
hwloc_obj_t _starpu_hip_get_hwloc_obj(hwloc_topology_t topology, int devid);
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

#pragma GCC visibility pop

#endif //  __DRIVER_HIP_H__

