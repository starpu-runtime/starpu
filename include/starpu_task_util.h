/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2014  Université de Bordeaux
 * Copyright (C) 2010-2013  Centre National de la Recherche Scientifique
 * Copyright (C) 2014       INRIA
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

#define STARPU_VALUE		 (1<<20)
#define STARPU_CALLBACK		 (2<<20)
#define STARPU_CALLBACK_WITH_ARG (3<<20)
#define STARPU_CALLBACK_ARG	 (4<<20)
#define STARPU_PRIORITY		 (5<<20)
#define STARPU_EXECUTE_ON_NODE	 (6<<20)
#define STARPU_EXECUTE_ON_DATA	 (7<<20)
#define STARPU_DATA_ARRAY        (8<<20)
#define STARPU_TAG               (9<<20)
#define STARPU_HYPERVISOR_TAG	 (10<<20)
#define STARPU_FLOPS	         (11<<20)
#define STARPU_SCHED_CTX	 (12<<20)
#define STARPU_PROLOGUE_CALLBACK   (13<<20)
#define STARPU_PROLOGUE_CALLBACK_ARG (14<<20)
#define STARPU_PROLOGUE_CALLBACK_POP   (15<<20)
#define STARPU_PROLOGUE_CALLBACK_POP_ARG (16<<20)
#define STARPU_EXECUTE_ON_WORKER (17<<20)
#define STARPU_TAG_ONLY          (18<<20)
#define STARPU_POSSIBLY_PARALLEL    (19<<20)
#define STARPU_WORKER_ORDER      (20<<20)

struct starpu_task *starpu_task_build(struct starpu_codelet *cl, ...);
int starpu_task_insert(struct starpu_codelet *cl, ...);
/* the function starpu_insert_task has the same semantics as starpu_task_insert, it is kept to avoid breaking old codes */
int starpu_insert_task(struct starpu_codelet *cl, ...);

void starpu_codelet_unpack_args(void *cl_arg, ...);

void starpu_codelet_pack_args(void **arg_buffer, size_t *arg_buffer_size, ...);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_TASK_UTIL_H__ */
