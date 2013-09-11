/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
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

#define STARPU_VALUE		 (1<<19)
#define STARPU_CALLBACK		 (1<<20)
#define STARPU_CALLBACK_WITH_ARG (1<<21)
#define STARPU_CALLBACK_ARG	 (1<<22)
#define STARPU_PRIORITY		 (1<<23)
#define STARPU_EXECUTE_ON_NODE	 (1<<24)
#define STARPU_EXECUTE_ON_DATA	 (1<<25)
#define STARPU_DATA_ARRAY        (1<<26)
#define STARPU_TAG               (1<<27)
#define STARPU_HYPERVISOR_TAG	 (1<<28)
#define STARPU_FLOPS	         (1<<29)
#define STARPU_SCHED_CTX	 (1<<30)
#define STARPU_PROLOGUE_CALLBACK   (1<<31)
#define STARPU_PROLOGUE_CALLBACK_ARG (1<<32)

int starpu_insert_task(struct starpu_codelet *cl, ...);

void starpu_codelet_unpack_args(void *cl_arg, ...);

void starpu_codelet_pack_args(void **arg_buffer, size_t *arg_buffer_size, ...);

#ifdef __cplusplus
}
#endif

#endif /* __STARPU_TASK_UTIL_H__ */
