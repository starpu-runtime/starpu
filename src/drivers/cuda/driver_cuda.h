/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2014,2016,2017                      Université de Bordeaux
 * Copyright (C) 2012,2017                                Inria
 * Copyright (C) 2015                                     Mathieu Lirzin
 * Copyright (C) 2010,2012,2015,2017                      CNRS
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

#include <common/config.h>

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#endif

#include <starpu.h>

extern struct _starpu_driver_ops _starpu_driver_cuda_ops;

void _starpu_cuda_init(void);
unsigned _starpu_get_cuda_device_count(void);
extern int _starpu_cuda_bus_ids[STARPU_MAXCUDADEVS+STARPU_MAXNUMANODES][STARPU_MAXCUDADEVS+STARPU_MAXNUMANODES];

#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
void _starpu_cuda_discover_devices (struct _starpu_machine_config *);
void _starpu_init_cuda(void);
void *_starpu_cuda_worker(void *);
#else
#  define _starpu_cuda_discover_devices(config) ((void) config)
#endif

#ifdef STARPU_USE_CUDA
cudaStream_t starpu_cuda_get_local_in_transfer_stream(void);
cudaStream_t starpu_cuda_get_in_transfer_stream(unsigned dst_node);
cudaStream_t starpu_cuda_get_local_out_transfer_stream(void);
cudaStream_t starpu_cuda_get_out_transfer_stream(unsigned src_node);
cudaStream_t starpu_cuda_get_peer_transfer_stream(unsigned src_node, unsigned dst_node);

#ifdef STARPU_USE_CUDA_MAP
uintptr_t _starpu_cuda_map_ram(void *src_ptr, unsigned src_node, unsigned dst_node, size_t size, int *ret);
int _starpu_cuda_unmap_ram(void *src_ptr, unsigned src_node, void *dst_ptr, unsigned dst_node, size_t size);
#endif
#endif

#endif //  __DRIVER_CUDA_H__

