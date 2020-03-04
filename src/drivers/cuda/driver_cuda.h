/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2015       Mathieu Lirzin
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

#ifdef STARPU_USE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#endif

#include <starpu.h>
#include <core/workers.h>
#include <datawizard/node_ops.h>

extern struct _starpu_driver_ops _starpu_driver_cuda_ops;
extern struct _starpu_node_ops _starpu_driver_cuda_node_ops;

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
#endif

unsigned _starpu_cuda_test_request_completion(struct _starpu_async_channel *async_channel);
void _starpu_cuda_wait_request_completion(struct _starpu_async_channel *async_channel);

int _starpu_cuda_copy_interface_from_cpu_to_cuda(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);
int _starpu_cuda_copy_interface_from_cuda_to_cuda(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);
int _starpu_cuda_copy_interface_from_cuda_to_cpu(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);

int _starpu_cuda_copy_data_from_cuda_to_cuda(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);
int _starpu_cuda_copy_data_from_cuda_to_cpu(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);
int _starpu_cuda_copy_data_from_cpu_to_cuda(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);

int _starpu_cuda_copy2d_data_from_cuda_to_cuda(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst, struct _starpu_async_channel *async_channel);
int _starpu_cuda_copy2d_data_from_cuda_to_cpu(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst, struct _starpu_async_channel *async_channel);
int _starpu_cuda_copy2d_data_from_cpu_to_cuda(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t blocksize, size_t numblocks, size_t ld_src, size_t ld_dst, struct _starpu_async_channel *async_channel);

int _starpu_cuda_copy3d_data_from_cuda_to_cuda(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t blocksize, size_t numblocks_1, size_t ld1_src, size_t ld1_dst, size_t numblocks_2, size_t ld2_src, size_t ld2_dst, struct _starpu_async_channel *async_channel);
int _starpu_cuda_copy3d_data_from_cuda_to_cpu(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t blocksize, size_t numblocks_1, size_t ld1_src, size_t ld1_dst, size_t numblocks_2, size_t ld2_src, size_t ld2_dst, struct _starpu_async_channel *async_channel);
int _starpu_cuda_copy3d_data_from_cpu_to_cuda(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t blocksize, size_t numblocks_1, size_t ld1_src, size_t ld1_dst, size_t numblocks_2, size_t ld2_src, size_t ld2_dst, struct _starpu_async_channel *async_channel);

int _starpu_cuda_is_direct_access_supported(unsigned node, unsigned handling_node);
uintptr_t _starpu_cuda_malloc_on_node(unsigned dst_node, size_t size, int flags);
void _starpu_cuda_free_on_node(unsigned dst_node, uintptr_t addr, size_t size, int flags);

#endif //  __DRIVER_CUDA_H__

