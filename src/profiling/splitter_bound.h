/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2022-2025  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __SPLITTER_BOUND_H__
#define __SPLITTER_BOUND_H__

#include <starpu.h>
#include <starpu_bound.h>
#include <core/jobs.h>

#pragma GCC visibility push(hidden)

#ifdef STARPU_RECURSIVE_TASKS
extern void _starpu_splitter_bound_record(struct _starpu_job *j);
extern void _starpu_splitter_bound_record_split(struct _starpu_job *j);
extern void _starpu_splitter_bound_delete(struct _starpu_job * j);
extern void _starpu_splitter_bound_delete_split(struct _starpu_job * j);
unsigned long _starpu_splitter_bound_get_nb_split(struct _starpu_job *j);
unsigned long _starpu_splitter_bound_get_nb_nsplit(struct _starpu_job *j);
extern double _starpu_splitter_bound_calculate();
void _starpu_splitter_bound_start();
#endif /* STARPU_RECURSIVE_TASKS */

#pragma GCC visibility pop

#endif // __SPLITTER_BOUND_H__
