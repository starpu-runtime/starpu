/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2015-2015  Mathieu Lirzin
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

#ifndef __DRIVER_CUDA_H__
#define __DRIVER_CUDA_H__

/** @file */

#include <common/config.h>

void _starpu_cuda_preinit(void);

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#ifdef STARPU_HAVE_NVML_H
#include <nvml.h>
#endif
#endif

#include <starpu.h>
#include <core/workers.h>
#include <datawizard/node_ops.h>

#pragma GCC visibility push(hidden)

extern struct _starpu_driver_ops _starpu_driver_cuda_ops;
extern struct _starpu_node_ops _starpu_driver_cuda_node_ops;

extern int _starpu_nworker_per_cuda;

#ifdef STARPU_HAVE_HWLOC
struct _starpu_machine_topology;
hwloc_obj_t _starpu_cuda_get_hwloc_obj(hwloc_topology_t topology, int devid);
#endif
extern int _starpu_cuda_bus_ids[STARPU_MAXCUDADEVS+STARPU_MAXNUMANODES][STARPU_MAXCUDADEVS+STARPU_MAXNUMANODES];

#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
void _starpu_cuda_early_init(void);
void _starpu_cuda_discover_devices (struct _starpu_machine_config *);
void _starpu_init_cuda_config(struct _starpu_machine_topology *topology, struct _starpu_machine_config *);
void _starpu_cuda_init_worker_binding(struct _starpu_machine_config *config, int no_mp_config, struct _starpu_worker *workerarg);
void _starpu_cuda_init_worker_memory(struct _starpu_machine_config *config, int no_mp_config, struct _starpu_worker *workerarg);
void _starpu_init_cuda(void);
void _starpu_init_cublas_v2_func(void);
void _starpu_shutdown_cublas_v2_func(void);
void _starpu_cublas_v2_init(void);
void _starpu_cublas_v2_shutdown(void);
void *_starpu_cuda_worker(void *);
#ifdef STARPU_HAVE_NVML_H
nvmlDevice_t _starpu_cuda_get_nvmldev(struct cudaDeviceProp *props);
extern __typeof__(nvmlInit) *_starpu_nvmlInit;
extern __typeof__(nvmlDeviceGetNvLinkState) *_starpu_nvmlDeviceGetNvLinkState;
extern __typeof__(nvmlDeviceGetNvLinkRemotePciInfo) *_starpu_nvmlDeviceGetNvLinkRemotePciInfo;
extern __typeof__(nvmlDeviceGetHandleByIndex) *_starpu_nvmlDeviceGetHandleByIndex;
extern __typeof__(nvmlDeviceGetHandleByPciBusId) *_starpu_nvmlDeviceGetHandleByPciBusId;
extern __typeof__(nvmlDeviceGetIndex) *_starpu_nvmlDeviceGetIndex;
extern __typeof__(nvmlDeviceGetPciInfo) *_starpu_nvmlDeviceGetPciInfo;
extern __typeof__(nvmlDeviceGetUUID) *_starpu_nvmlDeviceGetUUID;
#if HAVE_DECL_NVMLDEVICEGETTOTALENERGYCONSUMPTION
extern __typeof__(nvmlDeviceGetTotalEnergyConsumption) *_starpu_nvmlDeviceGetTotalEnergyConsumption;
#endif
#if HAVE_DECL_NVMLDEVICEGETNVLINKREMOTEDEVICETYPE
extern __typeof__(nvmlDeviceGetNvLinkRemoteDeviceType) *_starpu_nvmlDeviceGetNvLinkRemoteDeviceType;
typedef nvmlIntNvLinkDeviceType_t _starpu_nvmlIntNvLinkDeviceType_t;
#define _STARPU_NVML_NVLINK_DEVICE_TYPE_GPU NVML_NVLINK_DEVICE_TYPE_GPU
#define _STARPU_NVML_NVLINK_DEVICE_TYPE_IBMNPU NVML_NVLINK_DEVICE_TYPE_IBMNPU
#define _STARPU_NVML_NVLINK_DEVICE_TYPE_SWITCH NVML_NVLINK_DEVICE_TYPE_SWITCH
#define _STARPU_NVML_NVLINK_DEVICE_TYPE_UNKNOWN NVML_NVLINK_DEVICE_TYPE_UNKNOWN
#else
typedef unsigned _starpu_nvmlIntNvLinkDeviceType_t;
#define _STARPU_NVML_NVLINK_DEVICE_TYPE_GPU	0x00
#define _STARPU_NVML_NVLINK_DEVICE_TYPE_IBMNPU	0x01
#define _STARPU_NVML_NVLINK_DEVICE_TYPE_SWITCH	0x02
#define _STARPU_NVML_NVLINK_DEVICE_TYPE_UNKNOWN	0xFF
#endif
#endif

#else
#  define _starpu_cuda_discover_devices(config) ((void) config)
#endif

#pragma GCC visibility pop

#endif //  __DRIVER_CUDA_H__

