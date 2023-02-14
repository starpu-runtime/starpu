/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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

#ifndef __SOURCE_COMMON_H__
#define __SOURCE_COMMON_H__

/** @file */

#ifdef STARPU_USE_MP

#include <core/sched_policy.h>
#include <core/task.h>
#include <drivers/mp_common/mp_common.h>

#pragma GCC visibility push(hidden)

/* Array of structures containing all the informations useful to send
 * and receive informations with devices */
#ifdef STARPU_USE_MPI_MASTER_SLAVE
extern struct _starpu_mp_node *_starpu_src_nodes[STARPU_NARCH][STARPU_MAXMPIDEVS];
#endif

#ifdef STARPU_USE_TCPIP_MASTER_SLAVE
extern struct _starpu_mp_node *_starpu_src_nodes[STARPU_NARCH][STARPU_MAXTCPIPDEVS];
#endif

int _starpu_src_common_store_message(struct _starpu_mp_node *node, void * arg, int arg_size, enum _starpu_mp_command answer);

enum _starpu_mp_command _starpu_src_common_wait_completed_execution(struct _starpu_mp_node *node, int devid, void **arg, int * arg_size);

int _starpu_src_common_sink_nbcores(struct _starpu_mp_node *node, int *buf);

int _starpu_src_common_lookup(struct _starpu_mp_node *node, void (**func_ptr)(void), const char *func_name);

starpu_cpu_func_t _starpu_src_common_get_cpu_func_from_codelet(struct starpu_codelet *cl, unsigned nimpl);

void(* _starpu_src_common_get_cpu_func_from_job(const struct _starpu_mp_node *node STARPU_ATTRIBUTE_UNUSED, struct _starpu_job *j))(void);

struct _starpu_mp_node *_starpu_src_common_get_mp_node_from_memory_node(int memory_node);
uintptr_t _starpu_src_common_allocate(unsigned dst_node, size_t size, int flags);
void _starpu_src_common_free(unsigned dst_node, uintptr_t addr, size_t size, int flags);

uintptr_t _starpu_src_common_map(unsigned dst_node, uintptr_t addr, size_t size);
void _starpu_src_common_unmap(unsigned dst_node, uintptr_t addr, size_t size);

int _starpu_src_common_execute_kernel(struct _starpu_mp_node *node,
				      void (*kernel)(void), unsigned coreid,
				      enum starpu_codelet_type type,
				      int is_parallel_task, int cb_workerid,
				      starpu_data_handle_t *handles,
				      void **interfaces,
				      unsigned nb_interfaces,
				      void *cl_arg, size_t cl_arg_size, int detached);

int _starpu_src_common_copy_host_to_sink_sync(struct _starpu_mp_node *mp_node, void *src, void *dst, size_t size);

int _starpu_src_common_copy_sink_to_host_sync(struct _starpu_mp_node *mp_node, void *src, void *dst, size_t size);

int _starpu_src_common_copy_sink_to_sink_sync(struct _starpu_mp_node *src_node, struct _starpu_mp_node *dst_node, void *src, void *dst, size_t size);

int _starpu_src_common_copy_host_to_sink_async(struct _starpu_mp_node *mp_node, void *src, void *dst, size_t size, void *event);

int _starpu_src_common_copy_sink_to_host_async(struct _starpu_mp_node *mp_node, void *src, void *dst, size_t size, void *event);

int _starpu_src_common_copy_sink_to_sink_async(struct _starpu_mp_node *src_node, struct _starpu_mp_node *dst_node, void *src, void *dst, size_t size, void *event);

int _starpu_src_common_copy_data_host_to_sink(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);
int _starpu_src_common_copy_data_sink_to_host(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);
int _starpu_src_common_copy_data_sink_to_sink(uintptr_t src, size_t src_offset, unsigned src_node, uintptr_t dst, size_t dst_offset, unsigned dst_node, size_t size, struct _starpu_async_channel *async_channel);

void _starpu_src_common_init_switch_env(unsigned this);
void _starpu_src_common_workers_set(struct _starpu_worker_set * worker_set, int ndevices, struct _starpu_mp_node ** mp_node);

void _starpu_src_common_deinit(void);

#pragma GCC visibility pop

#endif /* STARPU_USE_MP */

#endif /* __SOURCE_COMMON_H__ */
