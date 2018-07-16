/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2014                                Inria
 * Copyright (C) 2010-2018                                CNRS
 * Copyright (C) 2010-2015, 2018                          Université de Bordeaux
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

#ifndef __STARPU_TASK_UTIL_H__
#define __STARPU_TASK_UTIL_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <starpu.h>

#ifdef __cplusplus
extern "C"
{
#endif

void starpu_create_sync_task(starpu_tag_t sync_tag, unsigned ndeps, starpu_tag_t *deps, void (*callback)(void *), void *callback_arg);

/* NOTE: when adding a value here, please make sure to update both
 * src/util/starpu_task_insert_utils.c (in two places) and
 * mpi/src/starpu_mpi_task_insert.c and mpi/src/starpu_mpi_task_insert_fortran.c */
#define STARPU_MODE_SHIFT	17
#define STARPU_VALUE		 (1<<STARPU_MODE_SHIFT)
#define STARPU_CALLBACK		 (2<<STARPU_MODE_SHIFT)
#define STARPU_CALLBACK_WITH_ARG (3<<STARPU_MODE_SHIFT)
#define STARPU_CALLBACK_ARG	 (4<<STARPU_MODE_SHIFT)
#define STARPU_PRIORITY		 (5<<STARPU_MODE_SHIFT)
#define STARPU_EXECUTE_ON_NODE	 (6<<STARPU_MODE_SHIFT)
#define STARPU_EXECUTE_ON_DATA	 (7<<STARPU_MODE_SHIFT)
#define STARPU_DATA_ARRAY        (8<<STARPU_MODE_SHIFT)
#define STARPU_DATA_MODE_ARRAY   (9<<STARPU_MODE_SHIFT)
#define STARPU_TAG               (10<<STARPU_MODE_SHIFT)
#define STARPU_HYPERVISOR_TAG	 (11<<STARPU_MODE_SHIFT)
#define STARPU_FLOPS	         (12<<STARPU_MODE_SHIFT)
#define STARPU_SCHED_CTX	 (13<<STARPU_MODE_SHIFT)
#define STARPU_PROLOGUE_CALLBACK   (14<<STARPU_MODE_SHIFT)
#define STARPU_PROLOGUE_CALLBACK_ARG (15<<STARPU_MODE_SHIFT)
#define STARPU_PROLOGUE_CALLBACK_POP   (16<<STARPU_MODE_SHIFT)
#define STARPU_PROLOGUE_CALLBACK_POP_ARG (17<<STARPU_MODE_SHIFT)
#define STARPU_EXECUTE_ON_WORKER (18<<STARPU_MODE_SHIFT)
#define STARPU_EXECUTE_WHERE     (19<<STARPU_MODE_SHIFT)
#define STARPU_TAG_ONLY          (20<<STARPU_MODE_SHIFT)
#define STARPU_POSSIBLY_PARALLEL    (21<<STARPU_MODE_SHIFT)
#define STARPU_WORKER_ORDER      (22<<STARPU_MODE_SHIFT)
#define STARPU_NODE_SELECTION_POLICY (23<<STARPU_MODE_SHIFT)
#define STARPU_NAME		 (24<<STARPU_MODE_SHIFT)
#define STARPU_CL_ARGS		(25<<STARPU_MODE_SHIFT)
#define STARPU_CL_ARGS_NFREE	(26<<STARPU_MODE_SHIFT)
#define STARPU_TASK_DEPS_ARRAY	(27<<STARPU_MODE_SHIFT)
#define STARPU_TASK_COLOR       (28<<STARPU_MODE_SHIFT)
#define STARPU_HANDLES_SEQUENTIAL_CONSISTENCY (29<<STARPU_MODE_SHIFT)
#define STARPU_SHIFTED_MODE_MAX (30<<STARPU_MODE_SHIFT)

int starpu_task_set(struct starpu_task *task, struct starpu_codelet *cl, ...);
struct starpu_task *starpu_task_build(struct starpu_codelet *cl, ...);
int starpu_task_insert(struct starpu_codelet *cl, ...);
/* the function starpu_insert_task has the same semantics as starpu_task_insert, it is kept to avoid breaking old codes */
int starpu_insert_task(struct starpu_codelet *cl, ...);

void starpu_task_insert_data_make_room(struct starpu_codelet *cl, struct starpu_task *task, int *allocated_buffers, int current_buffer, int room);
void starpu_task_insert_data_process_arg(struct starpu_codelet *cl, struct starpu_task *task, int *allocated_buffers, int *current_buffer, int arg_type, starpu_data_handle_t handle);
void starpu_task_insert_data_process_array_arg(struct starpu_codelet *cl, struct starpu_task *task, int *allocated_buffers, int *current_buffer, int nb_handles, starpu_data_handle_t *handles);
void starpu_task_insert_data_process_mode_array_arg(struct starpu_codelet *cl, struct starpu_task *task, int *allocated_buffers, int *current_buffer, int nb_descrs, struct starpu_data_descr *descrs);

void starpu_codelet_pack_args(void **arg_buffer, size_t *arg_buffer_size, ...);

struct starpu_codelet_pack_arg_data
{
	char *arg_buffer;
	size_t arg_buffer_size;
	size_t current_offset;
	int nargs;
};

void starpu_codelet_pack_arg_init(struct starpu_codelet_pack_arg_data *state);
void starpu_codelet_pack_arg(struct starpu_codelet_pack_arg_data *state, const void *ptr, size_t ptr_size);
void starpu_codelet_pack_arg_fini(struct starpu_codelet_pack_arg_data *state, void **cl_arg, size_t *cl_arg_size);

void starpu_codelet_unpack_args(void *cl_arg, ...);
void starpu_codelet_unpack_args_and_copyleft(void *cl_arg, void *buffer, size_t buffer_size, ...);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_TASK_UTIL_H__ */
