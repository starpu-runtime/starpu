/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __SORT_DATA_HANDLES_H__
#define __SORT_DATA_HANDLES_H__

/** @file */

#include <starpu.h>
#include <common/config.h>
#include <stdlib.h>

#include <stdarg.h>
#include <core/jobs.h>
#include <datawizard/coherency.h>
#include <datawizard/memalloc.h>

#pragma GCC visibility push(hidden)

/** To avoid deadlocks, we reorder the different buffers accessed to by the task
 * so that we always grab the rw-lock associated to the handles in the same
 * order. */
void _starpu_sort_task_handles(struct _starpu_data_descr descr[], unsigned nbuffers);

/** The reordering however puts alongside some different handles, just because
 * they have the same root. When avoiding to lock/acquire/load the same handle
 * several times, we need to keep looking among those.
 */
int _starpu_handles_same_root(starpu_data_handle_t dataA, starpu_data_handle_t dataB);

#pragma GCC visibility pop

#endif // SORT_DATA_HANDLES
