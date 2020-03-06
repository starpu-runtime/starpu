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

#define STARPU_MODE_SHIFT	16
#define STARPU_VALUE		 (1<<STARPU_MODE_SHIFT)
#define STARPU_CALLBACK		 (2<<STARPU_MODE_SHIFT)
#define STARPU_CALLBACK_WITH_ARG (3<<STARPU_MODE_SHIFT)
#define STARPU_CALLBACK_ARG	 (4<<STARPU_MODE_SHIFT)
#define STARPU_PRIORITY		 (5<<STARPU_MODE_SHIFT)
#define STARPU_EXECUTE_ON_NODE	 (6<<STARPU_MODE_SHIFT)
#define STARPU_EXECUTE_ON_DATA	 (7<<STARPU_MODE_SHIFT)
#define STARPU_DATA_ARRAY        (8<<STARPU_MODE_SHIFT)
#define STARPU_TAG               (9<<STARPU_MODE_SHIFT)
#define STARPU_HYPERVISOR_TAG	 (10<<STARPU_MODE_SHIFT)
#define STARPU_FLOPS	         (11<<STARPU_MODE_SHIFT)
#define STARPU_SCHED_CTX	 (12<<STARPU_MODE_SHIFT)
#define STARPU_PROLOGUE_CALLBACK   (13<<STARPU_MODE_SHIFT)
#define STARPU_PROLOGUE_CALLBACK_ARG (14<<STARPU_MODE_SHIFT)
#define STARPU_EXECUTE_ON_WORKER (15<<STARPU_MODE_SHIFT)
#define STARPU_SHIFTED_MODE_MAX (16<<STARPU_MODE_SHIFT)

struct starpu_task *starpu_task_build(struct starpu_codelet *cl, ...);
int starpu_insert_task(struct starpu_codelet *cl, ...);

void starpu_codelet_unpack_args(void *cl_arg, ...);

void starpu_codelet_pack_args(void **arg_buffer, size_t *arg_buffer_size, ...);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_TASK_UTIL_H__ */
