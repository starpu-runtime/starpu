/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

enum _starpu_mp_command _starpu_src_common_wait_command_sync(struct _starpu_mp_node *node, void ** arg, int* arg_size);
int _starpu_src_common_store_message(struct _starpu_mp_node *node, void * arg, int arg_size, enum _starpu_mp_command answer);

enum _starpu_mp_command _starpu_src_common_wait_completed_execution(struct _starpu_mp_node *node, int devid, void **arg, int * arg_size);

int _starpu_src_common_sink_nbcores(struct _starpu_mp_node *node, int *buf);

int _starpu_src_common_lookup(const struct _starpu_mp_node *node, void (**func_ptr)(void), const char *func_name);

int _starpu_src_common_allocate(const struct _starpu_mp_node *mp_node, void **addr, size_t size);

void _starpu_src_common_free(struct _starpu_mp_node *mp_node, void *addr);

int _starpu_src_common_execute_kernel(const struct _starpu_mp_node *node,
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

int _starpu_src_common_locate_file(char *located_file_name, size_t len,
				   const char *env_file_name, const char *env_mic_path,
				   const char *config_file_name, const char *actual_file_name,
				   const char **suffixes);

void _starpu_src_common_worker(struct _starpu_worker_set * worker_set, unsigned baseworkerid, struct _starpu_mp_node * node_set);

#if defined(STARPU_USE_MPI_MASTER_SLAVE) && !defined(STARPU_MPI_MASTER_SLAVE_MULTIPLE_THREAD)
void _starpu_src_common_init_switch_env(unsigned this);
void _starpu_src_common_workers_set(struct _starpu_worker_set * worker_set, int ndevices, struct _starpu_mp_node ** mp_node);
#endif

#endif /* STARPU_USE_MP */

#endif /* __SOURCE_COMMON_H__ */
