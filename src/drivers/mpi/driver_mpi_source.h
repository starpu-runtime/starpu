/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __DRIVER_MPI_SOURCE_H__
#define __DRIVER_MPI_SOURCE_H__

/** @file */

#include <drivers/mp_common/mp_common.h>
#include <starpu_mpi_ms.h>
#include <datawizard/node_ops.h>

#ifdef STARPU_USE_MPI_MASTER_SLAVE

extern struct _starpu_node_ops _starpu_driver_mpi_node_ops;

/** Array of structures containing all the informations useful to send
 * and receive informations with devices */
extern struct _starpu_mp_node *_starpu_mpi_ms_nodes[STARPU_MAXMPIDEVS];
struct _starpu_mp_node *_starpu_mpi_src_get_mp_node_from_memory_node(int memory_node);
struct _starpu_mp_node *_starpu_mpi_ms_src_get_actual_thread_mp_node();

unsigned _starpu_mpi_src_get_device_count();
void *_starpu_mpi_src_worker(void *arg);

void _starpu_mpi_source_init(struct _starpu_mp_node *node);
void _starpu_mpi_source_deinit(struct _starpu_mp_node *node);

int _starpu_mpi_src_allocate_memory(void ** addr, size_t size, unsigned memory_node);
void _starpu_mpi_source_free_memory(void *addr, unsigned memory_node);

int _starpu_mpi_copy_mpi_to_ram_sync(void *src, unsigned src_node, void *dst, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, size_t size);
int _starpu_mpi_copy_ram_to_mpi_sync(void *src, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst, unsigned dst_node, size_t size);
int _starpu_mpi_copy_sink_to_sink_sync(void *src, unsigned src_node, void *dst, unsigned dst_node, size_t size);

int _starpu_mpi_copy_mpi_to_ram_async(void *src, unsigned src_node, void *dst, unsigned dst_node STARPU_ATTRIBUTE_UNUSED, size_t size, void * event);
int _starpu_mpi_copy_ram_to_mpi_async(void *src, unsigned src_node STARPU_ATTRIBUTE_UNUSED, void *dst, unsigned dst_node, size_t size, void * event);
int _starpu_mpi_copy_sink_to_sink_async(void *src, unsigned src_node, void *dst, unsigned dst_node, size_t size, void * event);

int _starpu_mpi_copy_interface_from_mpi_to_cpu(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);
int _starpu_mpi_copy_interface_from_mpi_to_mpi(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);
int _starpu_mpi_copy_interface_from_cpu_to_mpi(starpu_data_handle_t handle, void *src_interface, unsigned src_node, void *dst_interface, unsigned dst_node, struct _starpu_data_request *req);

int _starpu_mpi_copy_data_from_mpi_to_cpu(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);
int _starpu_mpi_copy_data_from_mpi_to_mpi(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);
int _starpu_mpi_copy_data_from_cpu_to_mpi(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);

int _starpu_mpi_is_direct_access_supported(unsigned node, unsigned handling_node);
uintptr_t _starpu_mpi_malloc_on_node(unsigned dst_node, size_t size, int flags);
void _starpu_mpi_free_on_node(unsigned dst_node, uintptr_t addr, size_t size, int flags);

starpu_mpi_ms_kernel_t _starpu_mpi_ms_src_get_kernel_from_codelet(struct starpu_codelet *cl, unsigned nimpl);
void(* _starpu_mpi_ms_src_get_kernel_from_job(const struct _starpu_mp_node *node STARPU_ATTRIBUTE_UNUSED, struct _starpu_job *j))(void);

#endif /* STARPU_USE_MPI_MASTER_SLAVE */

#endif	/* __DRIVER_MPI_SOURCE_H__ */
