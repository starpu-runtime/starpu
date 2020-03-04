/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_TASK_INSERT_UTILS_H__
#define __STARPU_TASK_INSERT_UTILS_H__

/** @file */

#include <stdlib.h>
#include <stdarg.h>
#include <starpu.h>

typedef void (*_starpu_callback_func_t)(void *);

int _starpu_codelet_pack_args(void **arg_buffer, size_t *arg_buffer_size, va_list varg_list);
int _starpu_task_insert_create(struct starpu_codelet *cl, struct starpu_task *task, va_list varg_list);
int _fstarpu_task_insert_create(struct starpu_codelet *cl, struct starpu_task *task, void **arglist);

#endif // __STARPU_TASK_INSERT_UTILS_H__

