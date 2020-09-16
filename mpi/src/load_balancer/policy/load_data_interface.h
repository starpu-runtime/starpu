/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/** @file */

#ifndef __LOAD_DATA_INTERFACE_H
#define __LOAD_DATA_INTERFACE_H

/** interface for load_data */
struct load_data_interface
{
	/** Starting time of the execution */
	double start;
	/** Elapsed time until the start time and the time when event "launch a load
	 * balancing phase" is triggered */
	double elapsed_time;
	/** Current submission phase, i.e how many balanced steps have already
	 * happened so far. */
	int phase;
	/** Number of currently submitted tasks */
	int nsubmitted_tasks;
	/** Number of currently finished tasks */
	int nfinished_tasks;
	/** Task threshold to sleep the submission thread */
	int sleep_task_threshold;
	/** Task threshold to wake-up the submission thread */
	int wakeup_task_threshold;
	/** Ratio of submitted tasks to wait for completion before waking up the
	 * submission thread */
	double wakeup_ratio;
};

void load_data_data_register(starpu_data_handle_t *handle, unsigned home_node, int sleep_task_threshold, double wakeup_ratio);

int load_data_get_sleep_threshold(starpu_data_handle_t handle);
int load_data_get_wakeup_threshold(starpu_data_handle_t handle);
int load_data_get_current_phase(starpu_data_handle_t handle);
int load_data_get_nsubmitted_tasks(starpu_data_handle_t handle);
int load_data_get_nfinished_tasks(starpu_data_handle_t handle);

int load_data_inc_nsubmitted_tasks(starpu_data_handle_t handle);
int load_data_inc_nfinished_tasks(starpu_data_handle_t handle);

int load_data_next_phase(starpu_data_handle_t handle);

int load_data_update_elapsed_time(starpu_data_handle_t handle);
double load_data_get_elapsed_time(starpu_data_handle_t handle);

int load_data_update_wakeup_cond(starpu_data_handle_t handle);
int load_data_wakeup_cond(starpu_data_handle_t handle);

#define LOAD_DATA_GET_NSUBMITTED_TASKS(interface)	(((struct load_data_interface *)(interface))->nsubmitted_tasks)
#define LOAD_DATA_GET_SLEEP_THRESHOLD(interface)	(((struct load_data_interface *)(interface))->sleep_task_threshold)
#define LOAD_DATA_GET_WAKEUP_THRESHOLD(interface)	(((struct load_data_interface *)(interface))->wakeup_task_threshold)

#endif /* __LOAD_DATA_INTERFACE_H */
