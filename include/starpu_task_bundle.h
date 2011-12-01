/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010, 2011  Université de Bordeaux 1
 * Copyright (C) 2011  Télécom-SudParis
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

#include <starpu.h>

#ifndef __STARPU_TASK_BUNDLE_H__
#define __STARPU_TASK_BUNDLE_H__

#include <starpu_config.h>

#if ! defined(_MSC_VER)
#  include <pthread.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

struct starpu_task_bundle_entry
{
	struct starpu_task *task;
	struct starpu_task_bundle_entry *next;
};

/* The task bundle structure describes a list of tasks that should be scheduled
 * together whenever possible. */
struct starpu_task_bundle
{
	/* Mutex protecting the bundle */
#if defined(_MSC_VER)
	void *mutex;
#else
	pthread_mutex_t mutex;
#endif
	/* last worker previously assigned a task from the bundle (-1 if none) .*/
	int previous_workerid;
	/* list of tasks */
	struct starpu_task_bundle_entry *list;
	/* If this flag is set, the bundle structure is automatically free'd
	 * when the bundle is deinitialized. */
	int destroy;
	/* Is the bundle closed ? */
	int closed;
	/* TODO retain bundle (do not schedule until closed) */
};

/* Initialize a task bundle */
void starpu_task_bundle_init(struct starpu_task_bundle *bundle);

/* Deinitialize a bundle. In case the destroy flag is set, the bundle structure
 * is freed too. */
void starpu_task_bundle_deinit(struct starpu_task_bundle *bundle);

/* Insert a task into a bundle. */
int starpu_task_bundle_insert(struct starpu_task_bundle *bundle, struct starpu_task *task);

/* Remove a task from a bundle. This method must be called with bundle->mutex
 * hold. This function returns 0 if the task was found, -ENOENT if the element
 * was not found, 1 if the element is found and if the list was deinitialized
 * because it became empty. */
int starpu_task_bundle_remove(struct starpu_task_bundle *bundle, struct starpu_task *task);

/* Close a bundle. No task can be added to a closed bundle. A closed bundle
 * automatically gets deinitialized when it becomes empty. */
void starpu_task_bundle_close(struct starpu_task_bundle *bundle);

/* Return the expected duration of the entire task bundle in µs. */
double starpu_task_bundle_expected_length(struct starpu_task_bundle *bundle, enum starpu_perf_archtype arch, unsigned nimpl);
/* Return the time (in µs) expected to transfer all data used within the bundle */
double starpu_task_bundle_expected_data_transfer_time(struct starpu_task_bundle *bundle, unsigned memory_node);
/* Return the expected power consumption of the entire task bundle in J. */
double starpu_task_bundle_expected_power(struct starpu_task_bundle *bundle,  enum starpu_perf_archtype arch, unsigned nimpl);

#ifdef __cplusplus
}
#endif

#endif // __STARPU_TASK_BUNDLE_H__
