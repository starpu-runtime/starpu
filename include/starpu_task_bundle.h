/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011  Université de Bordeaux 1
 * Copyright (C) 2011  Télécom-SudParis
 * Copyright (C) 2012  Inria
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

#ifndef __STARPU_TASK_BUNDLE_H__
#define __STARPU_TASK_BUNDLE_H__

#include <starpu_perfmodel.h>

#ifdef __cplusplus
extern "C"
{
#endif

struct starpu_task;

/* starpu_task_bundle_t
 * ==================
 * Purpose
 * =======
 * Opaque structure describing a list of tasks that should be scheduled
 * on the same worker whenever it's possible.
 * It must be considered as a hint given to the scheduler as there is no guarantee that
 * they will be executed on the same worker.
 */
typedef struct _starpu_task_bundle *starpu_task_bundle_t;

/* Initialize a task bundle */
void starpu_task_bundle_init(starpu_task_bundle_t *bundle);

/* Deinitialize a bundle. In case the destroy flag is set, the bundle structure
 * is freed too. */
void starpu_task_bundle_deinit(starpu_task_bundle_t bundle);

/* Insert a task into a bundle. */
int starpu_task_bundle_insert(starpu_task_bundle_t bundle, struct starpu_task *task);

/* Remove a task from a bundle. This method must be called with bundle->mutex
 * hold. This function returns 0 if the task was found, -ENOENT if the element
 * was not found, 1 if the element is found and if the list was deinitialized
 * because it became empty. */
int starpu_task_bundle_remove(starpu_task_bundle_t bundle, struct starpu_task *task);

/* Close a bundle. No task can be added to a closed bundle. A closed bundle
 * automatically gets deinitialized when it becomes empty. */
void starpu_task_bundle_close(starpu_task_bundle_t bundle);

/* Return the expected duration of the entire task bundle in µs. */
double starpu_task_bundle_expected_length(starpu_task_bundle_t bundle, enum starpu_perf_archtype arch, unsigned nimpl);
/* Return the time (in µs) expected to transfer all data used within the bundle */
double starpu_task_bundle_expected_data_transfer_time(starpu_task_bundle_t bundle, unsigned memory_node);
/* Return the expected power consumption of the entire task bundle in J. */
double starpu_task_bundle_expected_power(starpu_task_bundle_t bundle, enum starpu_perf_archtype arch, unsigned nimpl);

#ifdef __cplusplus
}
#endif

#endif // __STARPU_TASK_BUNDLE_H__
